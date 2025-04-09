# https://colab.research.google.com/drive/15vNZb6AsU7byjYoaEtXuNu567JWNzXOz#scrollTo=58a8fbc1e477db48

import numpy as np
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

_tokenizer, _model = None, None
model_name = 'jinaai/jina-embeddings-v3'


def init():
    # load model and tokenizer
    global _tokenizer, _model
    if not _tokenizer:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
    if not _model:
        _model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True)
    return _model, _tokenizer


def chunk_by_sentences(input_text: str, tokenizer: callable = None):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use (optional, will use global tokenizer if not provided)
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    init()
    if not tokenizer:
        tokenizer = _tokenizer
    inputs = tokenizer(input_text, return_tensors='pt',
                       return_offsets_mapping=True)
    sentence_endings = ['.', '?', '!', '。', '？', '！']
    sentence_ending_ids = [
        tokenizer.convert_tokens_to_ids(se) for se in sentence_endings
    ]
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id in sentence_ending_ids and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    chunks = [
        input_text[x[1]: y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def pool_embeddings(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)
    return outputs


def chunking(input_text, tokenizer: callable = None):
    init()
    # determine chunks
    chunks, span_annotations = chunk_by_sentences(input_text, tokenizer)
    # print('Chunks:\n- "' + '"\n- "'.join(chunks) + '"')
    # chunk afterwards (context-sensitive chunked pooling)
    inputs = _tokenizer(input_text, return_tensors='pt')
    model_output = _model(**inputs)
    embeddings = pool_embeddings(model_output, [span_annotations])[0]
    return chunks, span_annotations, embeddings


def get_tokenizer():
    init()
    return _tokenizer


def get_model():
    init()
    return _model


def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


__all__ = [
    'chunk_by_sentences', 'pool_embeddings', 'chunking', 'get_tokenizer', 'get_model'
]


if __name__ == '__main__':
    init()
    input_text = """Berlin (/bɜːrˈlɪn/ bur-LIN; German: [bɛʁˈliːn] ⓘ)[10] is the capital and largest city of Germany, by both area and population.[11] With 3.7 million inhabitants,[5] it has the highest population within its city limits of any city in the European Union. The city is also one of the states of Germany, being the third smallest state in the country by area. Berlin is surrounded by the state of Brandenburg, and Brandenburg's capital Potsdam is nearby. The urban area of Berlin has a population of over 4.6 million and is therefore the most populous urban area in Germany.[6][12] The Berlin-Brandenburg capital region has around 6.2 million inhabitants and is Germany's second-largest metropolitan region after the Rhine-Ruhr region,[5] as well as the fifth-biggest metropolitan region by GDP in the European Union.[13]"""
    chunks, span_annotations, embeddings = chunking(input_text)
    embeddings_traditional_chunking = _model.encode(chunks)
    berlin_embedding = _model.encode('Berlin')
    for chunk, new_embedding, trad_embeddings in zip(chunks, embeddings, embeddings_traditional_chunking):
        print(f'similarity_new("Berlin", "{chunk}"):', cos_sim(
            berlin_embedding, new_embedding))
        print(f'similarity_trad("Berlin", "{chunk}"):', cos_sim(
            berlin_embedding, trad_embeddings))
