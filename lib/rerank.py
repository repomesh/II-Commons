import torch
from transformers import AutoModel

MODEL_NAME = 'jinaai/jina-reranker-m0'
model = None


def init():
    global model
    if not model:
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

        # comment out the flash_attention_2 line if you don't have a compatible GPU
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        model.to(device)
        model.eval()
    return model


def rerank(query, documents):
    init()
    text_pairs = [[query, doc] for doc in documents]
    return model.compute_score(text_pairs, max_length=8192, doc_type="text")


__all__ = [
    'init',
    'rerank',
]
