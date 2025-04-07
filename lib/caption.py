from lib.preprocess import process
from lib.s3 import download_file, get_url_by_key
from lib.utilitas import Image, json_dumps, sha256
from torch.utils.data import Dataset, DataLoader
from vllm import LLM, SamplingParams
from vllm import SamplingParams
import os
import re
import tempfile
import time
import base64
from io import BytesIO
from openai import OpenAI
from lib.api import CaptionAPI

MODEL_NAME = os.getenv('CAPTION_MODEL') or 'Qwen/Qwen2.5-VL-7B-Instruct'
SAMPLING_PARAMS = SamplingParams(
    temperature=0.1, max_tokens=512, stop_token_ids=None
)
BATCH_SIZE = 8  # 64

MAX_SIZE = 512

model = None
prompts = None
dummy_image = None


def load_model(model_name=MODEL_NAME):
    return LLM(
        model=model_name, max_model_len=7000, max_num_seqs=5,
        gpu_memory_utilization=0.97,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #    "min_pixels": 28 * 28,
        #    "max_pixels": 1024 * 28 * 28,
        # },
        disable_mm_preprocessor_cache=True, enforce_eager=True,
    )


def load_all_prompts(tmpl_dir='prompts', force_reload=False):
    global prompts
    if not prompts or force_reload:
        prompts = {}
        for filename in os.listdir(tmpl_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(tmpl_dir, filename)
                with open(filepath, 'r') as f:
                    prompts[os.path.splitext(filename)[0]] = f.read()
    return prompts


def build_prompt_input(images, prompt_name='image_question_qwen25'):
    prompt = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
              f'<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'
              f'{prompts[prompt_name]}<|im_end|>\n<|im_start|>assistant\n')
    return [{
        'prompt': prompt, 'multi_modal_data': {'image': image}
    } for image in images]


def img_collate(meta_items):
    task_hash = sha256(json_dumps(meta_items))
    temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
    images = []
    for meta in meta_items:
        s3_address = meta['processed_storage_id']
        url = get_url_by_key(s3_address)
        filename = os.path.join(temp_path.name, f"{sha256(s3_address)}.jpg")
        snapshot = f"[{meta['id']}] {url}"
        print(f'🖼️ Fetching image: {snapshot} => {filename}')
        try:
            download_file(s3_address, filename)
            ps_result = process(filename, max_size=MAX_SIZE, in_pil=True)
            images.append(
                {'id': meta['id'], 'image': ps_result['processed_image']}
            )
        except Exception as e:
            print(f'❌ ({snapshot}) {e}')
            images.append({'id': None, 'image': dummy_image})
    return images


def init(model_name=MODEL_NAME):
    global model
    global dummy_image
    model = load_model(model_name)
    dummy_image = Image.new('RGB', (MAX_SIZE, MAX_SIZE), color='black')
    load_all_prompts()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def extract_captions(text):
    long_match = re.search(
        r'^.*caption\.long:\s*(.*?)\s*$', text, re.MULTILINE)
    short_match = re.search(
        r'^.*caption\.short:\s*(.*?)\s*$', text, re.MULTILINE)
    long = long_match.group(1).strip('"').strip() if long_match else None
    short = short_match.group(1).strip('"').strip() if short_match else None
    return long, short


class ImageListDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def caption_image(image_list):
    global model
    if not model:
        init()
    dataloader = DataLoader(
        dataset=ImageListDataset(image_list),
        batch_size=len(image_list),
        shuffle=False,
        drop_last=False,
        num_workers=len(image_list),
        collate_fn=img_collate,
    )
    ids, imgs, result = [], [], {}
    for _, images in enumerate(dataloader):
        _ids, _imgs = zip(*((item['id'], item['image']) for item in images))
        ids.extend(_ids)
        imgs.extend(_imgs)
    inputs = build_prompt_input(imgs)
    start_time = time.time()
    outputs = model.generate(inputs, sampling_params=SAMPLING_PARAMS)
    end_time = time.time()
    assert len(outputs) == len(imgs), \
        f'❌ Failed: Mismatched output {len(imgs)} => {len(outputs)}'
    c, t = len(imgs), end_time - start_time
    r = t / c if c != 0 else 0
    print(f'📸 Caption {c} images in {t:.2f} seconds, {r:.2f} sec/img.')
    for i, o in enumerate(outputs):
        long, short = extract_captions(o.outputs[0].text)
        assert len(long) and len(short), \
            f'❌ Unable to extract captions: {image_list[i]}'
        result[ids[i]] = {'caption_long': long, 'caption': short}
    return result


def caption_image_api(image_list, api_endpoint="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=None, model_name="gemini-2.0-pro-exp-02-05", prompt_name='image_question_qwen25', max_retries=10, retry_wait_seconds=1):
    """
    Generate captions for a list of images using the VLLM API.

    Args:
        image_list: List of image metadata
        api_endpoint: API endpoint URL (required)
        api_key: API key for authentication (optional, depends on the API)
        model_name: Name of the VLLM model to use
        prompt_name: Name of the prompt template to use
        max_retries: Maximum number of retry attempts for API calls (default: 3)
        retry_wait_seconds: Wait time in seconds between retry attempts (default: 2)

    Returns:
        Dictionary of image IDs mapped to their captions
    """

    # Process images similar to caption_image
    dataloader = DataLoader(
        dataset=ImageListDataset(image_list),
        batch_size=len(image_list),
        shuffle=False,
        drop_last=False,
        num_workers=len(image_list),
        collate_fn=img_collate,
    )

    ids, imgs, result = [], [], {}
    for _, images in enumerate(dataloader):
        _ids, _imgs = zip(*((item['id'], item['image']) for item in images))
        ids.extend(_ids)
        imgs.extend(_imgs)

    # Initialize CaptionAPI client
    caption_api = CaptionAPI(
        api_endpoint=api_endpoint,
        api_key=api_key,
        model_name=model_name,
        max_retries=max_retries,
        retry_wait_seconds=retry_wait_seconds
    )

    # Process images using the CaptionAPI
    result = caption_api.batch_process_images(
        imgs=imgs,
        ids=ids,
        prompt=prompts[prompt_name],
        extract_captions_func=extract_captions
    )

    return result


__all__ = [
    'BATCH_SIZE',
    'init',
    'caption_image',
    'caption_image_api',
]
