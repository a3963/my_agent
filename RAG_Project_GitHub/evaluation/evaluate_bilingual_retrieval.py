import json
from pathlib import Path

from backend.rag.embedding import get_embeddings
from backend.rag.vectorstore import load_vectorstore
from backend.rag.retriever import search_document_bilingual



base_dir = Path(__file__).resolve().parent.parent

dataset_path = base_dir / 'evaluation' / 'eval_dataset.json'
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

k = 3

hit_count = 0
mrr_sum = 0


for item in dataset:

    question = item['question']
    relevant_source = item['relevant_source']

    english_question = item['english_query']

    print(
        '英文Query：',
        english_question
    )

    documents = search_document_bilingual(
        vectorstore,
        chinese_query=question,
        english_query=english_question,
        fetch_k=20,
        k=k
    )

    retrieved_sources = []

    for doc in documents:
        source = doc.metadata.get('source','')
        retrieved_sources.append(source)


    print('\n问题：',question)

    print('正确来源：',relevant_source)

    print('检索结果：')

    for index,source in enumerate(
        retrieved_sources,
        start=1
    ):
        print(index,source)


    if relevant_source in retrieved_sources:

        hit_count += 1

        rank = retrieved_sources.index(
            relevant_source
        ) + 1

        mrr_sum += 1 / rank

        print('检索成功，排名：',rank)

    else:

        print('检索失败')


hit_at_k = hit_count / len(dataset)

mrr = mrr_sum / len(dataset)


print('\n======================')

print('评测问题数量：',len(dataset))

print(
    f'Hit@{k}:',
    round(hit_at_k,4)
)

print(
    'MRR:',
    round(mrr,4)
)