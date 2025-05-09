import cv2
import numpy
import torch

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


def normalize(features):
    if isinstance(features, torch.Tensor):
        tensor = features
    else:
        processed = []
        dtypes = []
        for x in features:
            if isinstance(x, torch.Tensor):
                arr = x.cpu().numpy()
            elif hasattr(x, 'tolist'):
                arr = x.tolist()
            elif hasattr(x, 'numpy'):
                arr = x.numpy()
            elif hasattr(x, 'to_numpy'):
                arr = x.to_numpy()
            else:
                raise TypeError(f"Cannot convert {type(x)} to array. Please provide a .tolist() or .to_numpy() method.")
            arr = numpy.array(arr)
            dtypes.append(arr.dtype)
            processed.append(arr)
        if all(dt == numpy.float16 for dt in dtypes):
            target_dtype = numpy.float16
            torch_dtype = torch.float16
        else:
            target_dtype = numpy.float32
            torch_dtype = torch.float32
        processed = [numpy.array(a, dtype=target_dtype) for a in processed]
        tensor = torch.tensor(numpy.stack(processed), dtype=torch_dtype)
    return torch.nn.functional.normalize(tensor, p=2, dim=1).cpu().numpy()

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
