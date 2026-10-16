import json
import pathlib

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    FactualCorrectness
)

from backend.rag.embedding import get_embeddings
from backend.rag.vectorstore import load_vectorstore
from backend.rag.retriever import search_chunk_bilingual
from backend.rag.reranker import get_reranker,rerank_documents
from backend.rag.generator import create_llm,generate_answer


basedir = pathlib.Path(__file__).resolve().parent.parent

dataset_path = (
    basedir
    / 'evaluation'
    / 'eval_dataset.json'
)

persist_directory = (
    basedir
    / 'data'
    / 'chroma_db_m3'
)


with open(
        dataset_path,
        'r',
        encoding='utf-8'
) as f:

    dataset = json.load(f)


# 第一次先跑3题
dataset = dataset[:3]


# =========================
# 1. 加载你的RAG系统
# =========================

embedding_model = get_embeddings(
    'BAAI/bge-m3'
)

vectorstore = load_vectorstore(
    persist_directory,
    embedding_model
)

reranker = get_reranker()

llm = create_llm(
    temperature=0
)


# =========================
# 2. 创建RAGAS Judge LLM
# =========================

api_key = llm.openai_api_key.get_secret_value()

base_url = llm.openai_api_base

model_name = llm.model_name


if base_url is None:

    base_url = str(
        llm.root_client.base_url
    )


ragas_client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url
)


ragas_llm = llm_factory(
    model_name,
    provider='openai',
    client=ragas_client,
    temperature=0
)


# =========================
# 3. 创建RAGAS指标
# =========================

context_precision_metric = ContextPrecision(
    llm=ragas_llm
)

context_recall_metric = ContextRecall(
    llm=ragas_llm
)

faithfulness_metric = Faithfulness(
    llm=ragas_llm
)

factual_correctness_metric = FactualCorrectness(
    llm=ragas_llm
)


context_precision_scores = []
context_recall_scores = []
faithfulness_scores = []
factual_correctness_scores = []


# =========================
# 4. 开始评测
# =========================

for item in dataset:

    question = item['question']

    english_query = item['english_query']

    reference_answer = item[
        'reference_answer'
    ]


    # -------- Retriever --------

    candidate_documents = search_chunk_bilingual(
        vectorstore,
        chinese_query=question,
        english_query=english_query,
        fetch_k=50,
        k=20
    )


    # -------- Reranker --------

    documents = rerank_documents(
        reranker,
        chinese_query=question,
        english_query=english_query,
        documents=candidate_documents,
        k=5
    )


    # RAGAS要求contexts是字符串列表
    contexts = [
        doc.page_content
        for doc in documents
    ]


    # -------- Generator --------

    answer = generate_answer(
        llm=llm,
        query=question,
        documents=documents
    )


    print(
        '\n=========================='
    )

    print(
        '问题：',
        question
    )

    print(
        '\nRAG回答：'
    )

    print(
        answer
    )


    # =========================
    # Context Precision
    # =========================

    precision_result = (
        context_precision_metric.score(
            user_input=question,
            reference=reference_answer,
            retrieved_contexts=contexts
        )
    )


    # =========================
    # Context Recall
    # =========================

    recall_result = (
        context_recall_metric.score(
            user_input=question,
            reference=reference_answer,
            retrieved_contexts=contexts
        )
    )


    # =========================
    # Faithfulness
    # =========================

    faithfulness_result = (
        faithfulness_metric.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )
    )


    # =========================
    # Factual Correctness
    # =========================

    factual_result = (
        factual_correctness_metric.score(
            response=answer,
            reference=reference_answer
        )
    )


    context_precision = (
        precision_result.value
    )

    context_recall = (
        recall_result.value
    )

    faithfulness = (
        faithfulness_result.value
    )

    factual_correctness = (
        factual_result.value
    )


    context_precision_scores.append(
        context_precision
    )

    context_recall_scores.append(
        context_recall
    )

    faithfulness_scores.append(
        faithfulness
    )

    factual_correctness_scores.append(
        factual_correctness
    )


    print(
        '\nRAGAS评测：'
    )

    print(
        'Context Precision:',
        round(context_precision,4)
    )

    print(
        'Context Recall:',
        round(context_recall,4)
    )

    print(
        'Faithfulness:',
        round(faithfulness,4)
    )

    print(
        'Factual Correctness:',
        round(factual_correctness,4)
    )


# =========================
# 5. 平均分
# =========================

avg_context_precision = (
    sum(context_precision_scores)
    / len(context_precision_scores)
)

avg_context_recall = (
    sum(context_recall_scores)
    / len(context_recall_scores)
)

avg_faithfulness = (
    sum(faithfulness_scores)
    / len(faithfulness_scores)
)

avg_factual_correctness = (
    sum(factual_correctness_scores)
    / len(factual_correctness_scores)
)


print(
    '\n=========================='
)

print(
    '最终 RAGAS Evaluation'
)

print(
    '问题数量：',
    len(dataset)
)

print(
    'Context Precision平均分：',
    round(avg_context_precision,4)
)

print(
    'Context Recall平均分：',
    round(avg_context_recall,4)
)

print(
    'Faithfulness平均分：',
    round(avg_faithfulness,4)
)

print(
    'Factual Correctness平均分：',
    round(avg_factual_correctness,4)
)