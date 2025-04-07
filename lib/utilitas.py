from PIL import ExifTags, Image, ImageFile, ImageOps
from pillow_heif import register_heif_opener
from pymediainfo import MediaInfo
from scipy.spatial import ConvexHull
import base64
import cv2
import datetime
import hashlib
import json
import math
import numpy
import os
import requests
import tempfile
import types
import warnings


ASCII = 'ascii'
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
IMG_EXTS = 'jpg|jpeg|png|heif|heifs|heic|heics|avci|avcs|avif|hif|tiff|tif|webp|mask'
ELLIPSE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
TIMEOUT = (3, 30)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    + 'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
}


class JsonEncoder(json.JSONEncoder):
    # Reference:
    # https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    # https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py
    # numpy.float_, numpy.complex_, was removed in numpy 2.0
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,  # type: ignore
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float16, numpy.float32, numpy.float64)):  # type: ignore
            return float(obj)
        elif isinstance(obj, (numpy.complex64, numpy.complex128)):  # type: ignore
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (numpy.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (numpy.bool_)):
            return bool(obj)
        elif isinstance(obj, (numpy.void)) or isinstance(obj, (types.FunctionType)):
            return None
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif obj.__class__.__name__ == 'IFDRational':  # for exif
            return 0 if obj.numerator == 0 else float(obj)
        return json.JSONEncoder.default(self, obj)


class Empty:
    pass


def _void(*args):
    pass


def get_color(color, channel=3):
    if color is None or color == 'BLACK':
        return COLOR_BLACK if channel == 3 else 0
    elif color == 'WHITE':
        return COLOR_WHITE if channel == 3 else 255
    return color


def init_image(size, channel=3, color=None, dtype=numpy.uint8):
    img = numpy.zeros((size[1], size[0], channel), dtype)
    img[::] = get_color(color, channel=channel)  # (B, G, R)
    return img


def read_image(file, as_bgr=False, as_rgba=False):  # default as RGB
    image = ImageOps.exif_transpose(Image.open(file)).convert(
        'RGBA' if as_rgba else 'RGB'
    )
    image = numpy.array(image)
    return invert_color(image) if as_bgr else image


def invert_color(image):
    return image[..., ::-1]


def crop_image(img, boundary):
    (boundary_top, boundary_bottom, boundary_left, boundary_right) = boundary
    original_height, original_width = img.shape[:2]
    return img[
        max(boundary_top, 0):min(boundary_bottom, original_height),
        max(boundary_left, 0):min(boundary_right, original_width)
    ]


def resize_image(img, size):
    # we just need downsize image, use LANCZOS4/AREA for the best quality
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def reshape_image(
    img,
    crop=None,
    size=None, fit=True, fill=False, keep_ratio=True,
    padding=None, place_top_left=False, bleeding=False,
    expand=None, enhance=None
):
    # check input
    if fill:
        fit = False
    # crop
    if crop is not None:
        img = crop_image(img, crop)
    # resize
    # fit is true means the image will be resized to fit the size
    # fit is false means the image will be resized only if it's bigger
    original_height, original_width = img.shape[:2]
    if not (size is None or (fit == False
                             and original_width <= size[0]
                             and original_height <= size[1] and not fill)):
        width_ratio = size[0] / original_width
        height_ratio = size[1] / original_height
        if keep_ratio:
            if width_ratio > height_ratio if fill else width_ratio < height_ratio:
                scale = size[0] / original_width
            else:
                scale = size[1] / original_height
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        else:
            new_width, new_height = size
        if (original_height < new_height or original_width < new_width) and enhance is not None:
            # eh_scale =  math.ceil(max(new_height / original_height, new_width / original_width))
            img = enhance(img)
        img = resize_image(img, (new_width, new_height))
        if fill and (new_width != size[0] or new_height != size[1]):
            crop_top = max((new_height - size[1]) // 2, 0)
            crop_left = max((new_width - size[0]) // 2, 0)
            img = crop_image(img, (
                crop_top, crop_top + size[1], crop_left, crop_left + size[0]
            ))
        # padding
        if padding is not None:
            if padding == True or padding == 'BLACK':
                padding = COLOR_BLACK
            elif padding == 'WHITE':
                padding = COLOR_WHITE
            delta_w = size[0] - new_width
            delta_h = size[1] - new_height
            if place_top_left:
                delta_top, delta_left = 0, 0
            elif bleeding and crop is not None:
                delta_top, delta_left = min(int(
                    size[1] * (- crop[0] / (crop[1] - crop[0]))
                ), delta_h) if crop[0] < 0 else 0, 0
            else:
                delta_top, delta_left = delta_h // 2, delta_w // 2
            delta_bottom, delta_right = delta_h - delta_top, delta_w - delta_left
            img = cv2.copyMakeBorder(
                img, delta_top, delta_bottom, delta_left, delta_right,
                cv2.BORDER_CONSTANT, value=padding
            )
    # expand
    if expand is not None:
        original_height, original_width = img.shape[:2]
        (exp_top, exp_bottom, exp_left, exp_right, exp_color) = expand
        new_img = init_image(size=(
            original_width + exp_left + exp_right,
            original_height + exp_top + exp_bottom
        ), color=get_color(exp_color))
        new_img[
            exp_top:exp_top + original_height,
            exp_left:exp_left + original_width
        ] = img
        img = new_img
    return img


def blend(image, mask, options={}):
    # all supported options:
    options = {
        'erode_iterations': 0,
        'blur_pixels': 0,
        'blur_radius': 0,
        **options
    }
    # Reference:
    # https://gist.github.com/clungzta/b4bbb3e2aa0490b0cfcbc042184b0b4e
    # https://gist.github.com/garybradski/fabe4ce7ed5c042988b748780393370c
    # https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
    # https://note.nkmk.me/en/python-opencv-numpy-alpha-blend-mask/
    # https://stackoverflow.com/questions/32290096/python-opencv-add-alpha-channel-to-rgb-image
    # https://stackoverflow.com/questions/55066764/how-to-blur-feather-the-edges-of-an-object-in-an-image-using-opencv
    # https://stackoverflow.com/questions/55969276/feather-cropped-edges
    # https://stackoverflow.com/questions/72215748/how-to-extend-mask-region-true-by-1-or-2-pixels
    # https://www.projectpro.io/recipes/print-full-numpy-array-without-truncating
    # feather mask
    erode = 0
    for key in ['erode_iterations', 'blur_pixels']:
        if options.get(key) is not None and options[key] > 0:
            erode += options[key]
    if erode > 0:
        mask = cv2.erode(mask, ELLIPSE_3, iterations=erode)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    if options.get('blur_radius') is not None and options['blur_radius'] > 0:
        if options['blur_radius'] % 2 == 0:
            options['blur_radius'] += 1
        mask = cv2.GaussianBlur(mask, (
            options['blur_radius'], options['blur_radius']
        ), 0)
    if options.get('invert_mask'):
        mask = cv2.bitwise_not(mask)
    # build background
    if options.get('background') is None:
        background = numpy.zeros(image.shape, numpy.uint8)
        background[::] = COLOR_WHITE  # (B, G, R)
    else:
        background = options.get('background')
    # blend
    mask_float = mask.astype('float') / 255.
    image_float = image.astype('float') / 255.
    background_float = background.astype('float') / 255.  # type: ignore
    result = background_float * (1 - mask_float) + image_float * mask_float
    result = (result * 255).astype('uint8')
    return result


def tweak_mask(mask):
    # Reference:
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#morphological-gradient
    resp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ELLIPSE_3)
    return cv2.morphologyEx(resp, cv2.MORPH_CLOSE, ELLIPSE_3)


# use white because it's common and to support mask blending
def rotate_image(img, angle, center=None, background=COLOR_WHITE, lossless=False):
    if lossless:
        angle = lossless_rotate_angle(angle)
    if angle == 0:
        return img
    elif center is None:
        if angle == 90 or angle == -270:
            img = cv2.transpose(img)
            return cv2.flip(img, 0)
        elif angle == -90 or angle == 270:
            img = cv2.transpose(img)
            return cv2.flip(img, 1)
        elif abs(angle) == 180:
            return cv2.flip(img, -1)
    (height, width) = img.shape[:2]
    center = (width // 2, height // 2) if center is None else center
    mtx = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(
        img, mtx, (width, height), borderValue=background
    )
    return img


def lossless_rotate_angle(angle):
    angle = angle % 360
    abs_angle = abs(angle)
    pos_ngt = angle / abs_angle if abs_angle != 0 else 0
    angle = sorted([
        [abs_angle, 0],
        [abs(abs_angle - 90), 90 * pos_ngt],
        [abs(abs_angle - 180), 180 * pos_ngt],
        [abs(abs_angle - 270), 270 * pos_ngt],
        [abs(abs_angle - 360), 0]
    ], key=lambda x: x[0])[0][1]
    return angle if abs(angle) > 0 else 0


def write_image(data, filename, format=None, invert=True, create_dir=True):
    if create_dir:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with_color = len(data.shape) == 3
    img = invert_color(data) if invert and with_color else data
    return cv2.imwrite(filename, img) if format is None \
        else cv2.imencode(format, img)[1].tofile(filename)


def read_file(filename, encoding='utf8'):
    return open(filename, 'r', encoding=encoding)


def read_json(filename):
    return json.load(read_file(filename))


def write_file(data, filename, encoding='utf8', create_dir=True):
    if create_dir:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w+', encoding=encoding) as file:
        file.write(data)


def write_json(data, filename):
    return write_file(json_dumps(data), filename)


def write_jsonl(data, filename, append=False):
    with open(filename, 'a' if append else 'w+', encoding='utf-8') as file:
        for item in data:
            file.write(json_dumps(item) + '\n')


def json_dumps(data,  indent=4, sort_keys=True, cls=JsonEncoder, compact=True):
    """
    Dump data to JSON string
    Args:
        data: Data to serialize
        indent: Indentation level (None for single line)
        sort_keys: Sort dictionary keys
        cls: JSON encoder class
        compact: If True, outputs single line
    """
    return json.dumps(
        data,
        indent=None if compact else indent,
        sort_keys=sort_keys,
        cls=cls,
        separators=(',', ':') if compact else None
    )


def jsonable(data):
    return json.loads(json_dumps(data))


def base64_bytes_encode(bytes):
    return base64.b64encode(bytes).decode(ASCII)


def base64_bytes_decode(string):
    return base64.decodebytes(bytes(string, ASCII))


def base64_numpy16_encode(np_array):
    return base64_bytes_encode(bytes(numpy.float16(np_array)))


def base64_numpy16_decode(string):
    return numpy.frombuffer(base64_bytes_decode(string), dtype=numpy.float16)


def get_2d_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_hull_by_points(points, img=None, background=COLOR_WHITE, color=COLOR_BLACK):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    if img is None:
        return hull_points
    mask = init_image(img.shape[:2][::-1], color=background)
    cv2.fillPoly(mask, numpy.array(
        [hull_points], dtype=numpy.int32), color)  # type: ignore
    return hull_points, mask

# Need to tweak this function
# def overlay_image(background, overlay, x, y, overlay_size=None):
#     """
#     @brief      Overlays a transparant PNG onto another image using CV2
#     @param      background_img    The background image
#     @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
#     @param      x                 x location to place the top-left corner of our overlay
#     @param      y                 y location to place the top-left corner of our overlay
#     @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
#     @return     Background image with overlay on top
#     """
#     result = background.copy()
#     if result.shape[2] == 3:
#         result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
#     if overlay_size is not None:
#         overlay = cv2.resize(overlay.copy(), overlay_size)
#     h, w, _ = overlay.shape
#     alpha_overlay = overlay[0:h, 0:w, 3] / 255.0
#     alpha_background = 1 - alpha_overlay
#     for c in range(0, 3):
#         result[y:y + h, x:x + w, c] = alpha_background * \
#             result[y: y + h, x:x + w, c] + alpha_overlay * overlay[0:h, 0:w, c]
#     return result

# use pymediainfo to detect file format


def media_info(file):
    media_info = MediaInfo.parse(file)
    return media_info.to_data()


def extract_exif(file):
    exif = {}
    image = Image.open(file)
    for tag, value in image.getexif().items():
        if tag in ExifTags.TAGS:
            if value.__class__.__name__ == 'bytes':
                for encoding in ['utf-16', 'utf-8', 'latin1']:
                    try:
                        value = value.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        pass
            if isinstance(value, str):
                value = value.replace('\u0000', '')
            exif[ExifTags.TAGS[tag]] = value
    return None if len(exif.keys()) == 0 else exif


def inspect_file(file, raw=False):
    media, exif = media_info(file), None
    try:
        exif = extract_exif(file)
    except Exception as e:
        pass
    return {'media': media, 'exif': exif if raw else jsonable(exif)}


def sha256(url):
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


def fetch(url, timeout=TIMEOUT, raw=False):  # (connect, transfer)
    if not url:
        raise ValueError('URL is required.')
    try:
        response = requests.get(
            url, stream=True, timeout=timeout, headers=HEADERS)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        return response if raw else response.content
    except Exception as e:
        raise RuntimeError(f"Fetching failed '{url}': {e}")


def download(url, filename=None, suffix=None, timeout=TIMEOUT):
    response = fetch(url, timeout, raw=True)
    hash = sha256(url)
    try:  # Save file to the directory
        if not filename:
            filename = os.path.join(
                tempfile.gettempdir(), 'utilitas',
                f'{hash}' + (f'.{suffix}' if suffix else '')
            )
        folder = os.path.dirname(filename)
        os.makedirs(folder, exist_ok=True)
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return filename
    except Exception as e:
        raise RuntimeError(f"Download failed '{url}': {e}")


def get_file_type(filename):
    try:
        with open(filename, 'rb') as f:
            if f.read(4).startswith(b'%PDF'):
                return 'PDF'
        with open(filename, 'r', encoding='utf-8') as f:
            f.read()
        return 'TEXT'
    except UnicodeDecodeError:
        pass
    return None


# initialize HEIF support
register_heif_opener()

# disable image size limit
Image.MAX_IMAGE_PIXELS = None

# disable image truncation limit
ImageFile.LOAD_TRUNCATED_IMAGES = True

# disable TIFF meta warnings
warnings.filterwarnings('ignore', message=r'.*had too many entries*')
warnings.filterwarnings('ignore', message=r'.*Truncated File Read*')

__all__ = [
    'COLOR_BLACK',
    'COLOR_WHITE',
    'ELLIPSE_3',
    'Empty',
    'IMG_EXTS',
    'Image',
    'JsonEncoder',
    '_void',
    'base64_bytes_decode',
    'base64_bytes_encode',
    'base64_numpy16_decode',
    'base64_numpy16_encode',
    'download',
    'extract_exif',
    'fetch',
    'get_2d_distance',
    'get_color',
    'get_hull_by_points',
    'init_image',
    'inspect_file',
    'invert_color',
    'json_dumps',
    'jsonable',
    'lossless_rotate_angle',
    'read_file',
    'read_image',
    'read_json',
    'reshape_image',
    'resize_image',
    'rotate_image',
    'sha256',
    'tweak_mask',
    'write_file',
    'write_image',
    'write_json',
    'write_jsonl',
    'get_file_type',
]
