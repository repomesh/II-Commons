from sentence_transformers import SentenceTransformer
import nltk
import re

MODEL = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
model, nltkReady = None, False


def init():
    global nltkReady
    nltk.download('punkt_tab')
    nltkReady = True
    global model
    model = SentenceTransformer(MODEL, trust_remote_code=True)
    model.max_seq_length = 8192


def chunk(document, size=2048, overlap=1, join=True):
    global nltkReady
    if not nltkReady:
        init()
    chunks, maxLen = [], int(size / 3)
    sentences = nltk.sent_tokenize(document)
    for i in range(len(sentences)):
        subs = []
        if len(sentences[i]) > maxLen:
            for word in re.split(r'\s+', sentences[i]):
                if len(subs) and len(subs[-1]) + 1 + len(word) < maxLen:
                    subs[-1] += f' {word}'
                else:
                    subs.append(word)
        else:
            subs.append(sentences[i])
        for j in range(len(subs)):
            if len(chunks) and len(' '.join(chunks[-1])) + len(subs[j]) < size:
                chunks[-1].append(subs[j])
            else:
                chunks.append((
                    chunks[-1][-overlap:] if len(chunks) else []
                ) + [subs[j]])
    return [' '.join(chunk) for chunk in chunks] if join else chunks


def encode(chunks):
    global model
    if not model:
        init()
    return model.encode(chunks).tolist()


def process(document, size=2048, overlap=1):
    try:
        chunks = chunk(document, size, overlap)
        embeddings = encode(chunks)
        result = []
        for i in range(len(chunks)):
            result.append({'chunk': chunks[i], 'embedding': embeddings[i]})
        return result
    except Exception as e:
        print(f'Error processing document: {e}')
        return None


__all__ = [
    'init',
    'chunk',
]
