from sentence_transformers import SentenceTransformer
import nltk
import re

# https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0
# MODEL = 'Snowflake/snowflake-arctic-embed-l-v2.0'
MODEL = 'Snowflake/snowflake-arctic-embed-m-v2.0'
model, nltkReady = None, False


def init():
    global nltkReady, model
    if not nltkReady:
        nltk.download('punkt_tab')
        nltkReady = True
    if not model:
        model = SentenceTransformer(MODEL, trust_remote_code=True)


def chunk_by_sentence(document):
    init()
    return nltk.sent_tokenize(document)


def chunk(document, size=2048, overlap=1, join=True):
    chunks, maxLen = [], int(size / 3)
    sentences = chunk_by_sentence(document)
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


def encode(chunks, query=False):
    init()
    resp = None
    if query:
        resp = model.encode(chunks, prompt_name='query')
    else:
        resp = model.encode(chunks)
    return resp.tolist()


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


if __name__ == '__main__':
    resp = process('Hello, world!')
    print(resp)
__all__ = [
    'init',
    'chunk',
    'chunk_by_sentence',
    'process',
]
