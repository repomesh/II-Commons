from huggingface_hub import hf_hub_download
from lib.utilitas import reshape_image
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPProcessor
import onnxruntime
import time
import torch


MODEL_NAME = 'jinaai/jina-clip-v1'
Q4_MODEL_TEXT = 'onnx/text_model_q4.onnx'
Q4_MODEL_IMAGE = 'onnx/vision_model_q4.onnx'
Q4_PROCESSOR = 'openai/clip-vit-base-patch32'
BATCH_SIZE = 100
MAX_SIZE = 384
MAX_IMAGE_SIZE = (MAX_SIZE, MAX_SIZE)
DIMENSION = 768

encoder = None


class Encoder:
    def __init__(self, model_name: str, q4=False):
        if q4:
            print("Using Q4 for model inference.")
            text_model_path, image_model_path = \
                hf_hub_download(repo_id=MODEL_NAME, filename=Q4_MODEL_TEXT), \
                hf_hub_download(repo_id=MODEL_NAME, filename=Q4_MODEL_IMAGE)
            if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                print('✅ Using GPU for ONNX Runtime')
                providers = ["CUDAExecutionProvider"]
            else:
                print('⚠️ GPU not available, using CPU')
                providers = ["CPUExecutionProvider"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            self.q4_processor = CLIPProcessor.from_pretrained(Q4_PROCESSOR)
            self.q4_text_session = onnxruntime.InferenceSession(
                text_model_path, providers=providers
            )
            self.q4_image_session = onnxruntime.InferenceSession(
                image_model_path, providers=providers
            )
        else:
            print("Using FP16 for model inference.")
            # Initialize the model (AutoModel from transformers instead of SentenceTransformer)
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float16
            ).to(device).eval()
            print(f"Loaded model `{model_name}`' to {device}.")
            self.model = torch.compile(self.model, mode='max-autotune')

    def encode_text(self, text: list[str]) -> list[float]:
        # Generate embeddings for text only
        if hasattr(self, 'tokenizer') and hasattr(self, 'q4_text_session'):
            text_tensor = self.tokenizer(
                text, return_tensors='np'
            )['input_ids']
            input_name = self.q4_text_session.get_inputs()[0].name
            output_name = self.q4_text_session.get_outputs()[0].name
            text_emb = self.q4_text_session.run(
                [output_name], {input_name: text_tensor}
            )[0]
        else:
            with torch.no_grad():
                text_emb = self.model.encode_text(text)
        return text_emb

    def encode_image(self, images: list, batch_size=BATCH_SIZE) -> list[list[float]]:
        # Generate embeddings for images only
        if hasattr(self, 'q4_processor') and hasattr(self, 'q4_image_session'):
            inputs = self.q4_processor(images=images, return_tensors='np')
            image_tensor = inputs['pixel_values']
            input_name = self.q4_image_session.get_inputs()[0].name
            output_name = self.q4_image_session.get_outputs()[0].name
            image_emb = self.q4_image_session.run(
                [output_name], {input_name: image_tensor}
            )[0]
        else:
            with torch.no_grad():
                image_emb = self.model.encode_image(
                    images, batch_size=batch_size
                )
        return image_emb


def prepare_image(img):
    return Image.fromarray(reshape_image(img, size=MAX_IMAGE_SIZE, fit=False))


def init(model_name=MODEL_NAME, q4=False):
    global encoder
    encoder = Encoder(model_name, q4=q4)


def encode_text(texts):
    global encoder
    if not encoder:
        init()
    start_time = time.time()
    text_embeddings = encoder.encode_text(texts)
    end_time = time.time()
    c, t = len(texts), end_time - start_time
    r = t / c if c != 0 else 0
    print(f'Encoded {c} texts in {t:.2f} seconds, {r:.2f} sec/txt.')
    return text_embeddings


def encode_image(imgs):
    global encoder
    if not encoder:
        init()
    start_time = time.time()
    images = [prepare_image(img) for img in imgs]
    image_embeddings = encoder.encode_image(images, batch_size=BATCH_SIZE)
    end_time = time.time()
    c, t = len(images), end_time - start_time
    r = t / c if c != 0 else 0
    print(f'Encoded {c} images in {t:.2f} seconds, {r:.2f} sec/img.')
    return image_embeddings


__all__ = [
    'init',
    'BATCH_SIZE',
    'DIMENSION',
    'encode_image',
    'encode_text',
]
