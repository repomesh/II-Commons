from lib.utilitas import Image, inspect_file, read_image, reshape_image
import cv2
import os
import sys

MIN_SIZE = 256
MAX_SIZE = 7680
ACCEPTED_FORMATS = ['JPEG']
# MAX_SIZE = 580 // for test only


def process(file, max_size=MAX_SIZE, in_pil=False):
    ins_resp = inspect_file(file)
    media = ins_resp.get('media')
    exif = ins_resp.get('exif')
    type = media['tracks'][0].get('format', '[UNABLE_TO_IDENTIFY]').upper()
    match type:
        case 'JPEG' | 'JPEG 2000' | 'PNG' | 'GIF' | 'WEBP' | 'TIFF' | 'LATM' \
                | '[UNABLE_TO_IDENTIFY]':
            try:
                processed_image = origin_image = read_image(file)
            except Exception as e:
                print(media)
                raise Exception(f'Not an image file, {e}')
            # debug: {
            # print(media)
            # }
            processed_width = origin_width = origin_image.shape[1]
            processed_height = origin_height = origin_image.shape[0]
            aspect_ratio = origin_width / origin_height if origin_height != 0 else 0
            size_str = f': {origin_width} * {origin_height}'
            processed = False
            if min(origin_width, origin_height) < MIN_SIZE:
                raise Exception(f'Image too small, dropped{size_str}')
            if max(origin_width, origin_height) > max_size:
                print(f'Image too large, resizing{size_str}')
                processed_image = reshape_image(
                    origin_image, size=(max_size, max_size), fit=False
                )
                processed_height, processed_width = processed_image.shape[:2]
                processed = True
            if type not in ACCEPTED_FORMATS:
                print(f'Reencoding image, origin format: {type}')
                processed = True
            if processed_image.shape[2] == 4:
                print(f'Converting RGBA to RGB...')
                processed_image = cv2.cvtColor(
                    processed_image, cv2.COLOR_RGBA2RGB
                )
                processed = True
            return {
                'processed_image': Image.fromarray(processed_image) if in_pil else processed_image,
                'processed': processed,
                'meta': {
                    'aspect_ratio': aspect_ratio,
                    'origin_height': origin_height,
                    'origin_width': origin_width,
                    'processed_height': processed_height,
                    'processed_width': processed_width,
                    'exif': exif or {},
                }
            }
        case 'MPEG AUDIO' | 'FLIC' | 'LATM' | 'SMPTE ST 337' | 'ADTS':
            # print(media)
            raise Exception(f'Unsupported format, dropped: {type}')
        case _:
            message = f'Unknown format, dropped: {type}'
            print(media)
            if os.environ.get('DEBUG') == 'true':
                print(message)
                sys.exit(1)
            raise Exception(message)


__all__ = [
    'process',
]
