from lib.utilitas import reshape_image
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModel
import time
import torch
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# https://huggingface.co/google/siglip2-so400m-patch16-naflex
# pip install git+https://github.com/huggingface/transformers.git
MODEL_NAME = 'google/siglip2-so400m-patch16-naflex'
BATCH_SIZE = 30
MAX_SIZE = 384
MAX_IMAGE_SIZE = (MAX_SIZE, MAX_SIZE)
DIMENSION = 768

model, processor, tokenizer, device = None, None, None, None


def prepare_image(img):
    return Image.fromarray(reshape_image(img, size=MAX_IMAGE_SIZE, fit=False))


def init(model_name=MODEL_NAME):
    global model
    global processor
    global tokenizer
    global device
    if torch.cuda.is_available():
        print('> 🐧 Using CUDA...')
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        # https://github.com/pytorch/pytorch/issues/77764
        print('>  Using MPS...')
        device = torch.device('mps')
    else:
        print('> ⚠️ Using CPU...')
        device = torch.device('cpu')
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def encode_text(texts, raw=False):
    global model
    if not model:
        init()
    start_time = time.time()
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=64, return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    end_time = time.time()
    c, t = len(texts), end_time - start_time
    r = t / c if c != 0 else 0
    print(f'Encoded {c} texts in {t:.2f} seconds, {r:.2f} sec/txt.')
    return text_features if raw else text_features.cpu().numpy()


def encode_image(imgs, raw=False):
    global model
    if not model:
        init()
    start_time = time.time()
    images = [prepare_image(img) for img in imgs]
    inputs = processor(images=images, return_tensors='pt').to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    end_time = time.time()
    c, t = len(images), end_time - start_time
    r = t / c if c != 0 else 0
    print(f'Encoded {c} images in {t:.2f} seconds, {r:.2f} sec/img.')
    return image_features if raw else image_features.cpu().numpy()


__all__ = [
    'init',
    'MODEL_NAME',
    'BATCH_SIZE',
    'MAX_SIZE',
    'MAX_IMAGE_SIZE',
    'DIMENSION',
    'encode_image',
    'encode_text',
]
