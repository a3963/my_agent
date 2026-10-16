import json
from pathlib import Path

from backend.rag.embedding import get_embeddings
from backend.rag.vectorstore import load_vectorstore
from backend.rag.retriever import search_chunk_bilingual
import re


def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r'[\s\-\u00ad\x00-\x1f]+',
        '',
        text
    )

    return text


base_dir = Path(__file__).resolve().parent.parent

dataset_path = base_dir / 'evaluation' / 'eval_chunk_dataset.json'

persist_directory = base_dir / 'data' / 'chroma_db_m3'


with open(dataset_path,'r',encoding='utf-8') as f:
    dataset = json.load(f)


embedding_model = get_embeddings(
    "BAAI/bge-m3"
)

vectorstore = load_vectorstore(
    persist_directory,
    embedding_model
)


k = 5

hit_count = 0
mrr_sum = 0


for item in dataset:

    question = item['question']
    english_query = item['english_query']
    relevant_source = item['relevant_source']
    relevant_keywords = item['relevant_keywords']


    documents = search_chunk_bilingual(
        vectorstore,
        chinese_query=question,
        english_query=english_query,
        fetch_k=30,
        k=k
    )


    hit_rank = None


    print('\n问题：',question)

    for index,doc in enumerate(
        documents,
        start=1
    ):

        source = doc.metadata.get(
            'source',
            ''
        )

        page = doc.metadata.get(
            '页数',
            ''
        )

        content = normalize_text(
            doc.page_content
        )

        print(
            index,
            source,
            '页码：',
            page
        )

        print(
            doc.page_content[:150]
        )

        if source == relevant_source:

            matched_keywords = 0

            for keyword in relevant_keywords:

                keyword = normalize_text(
                    keyword
                )

                if keyword in content:
                    matched_keywords += 1

            if matched_keywords >= 2:
                hit_rank = index
                break


    if hit_rank is not None:

        hit_count += 1

        mrr_sum += 1 / hit_rank

        print(
            '正确Chunk排名：',
            hit_rank
        )

    else:

        print(
            '正确Chunk未进入Top',
            k
        )


hit_at_k = hit_count / len(dataset)

mrr = mrr_sum / len(dataset)


print('\n======================')

print(
    'Chunk评测问题数量：',
    len(dataset)
)

print(
    f'Chunk Hit@{k}:',
    round(hit_at_k,4)
)

print(
    'Chunk MRR:',
    round(mrr,4)
)

#Chunk Hit@5 = 0.6667
#Chunk MRR   = 0.4444