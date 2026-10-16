import json
import pathlib

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from backend.rag.embedding import get_embeddings
from backend.rag.vectorstore import load_vectorstore
from backend.rag.retriever import search_chunk_bilingual
from backend.rag.reranker import get_reranker, rerank_documents
from backend.rag.generator import create_llm, generate_answer


# =========================================
# 1. 定义Pydantic评测结果结构
# =========================================

class RAGEvaluation(BaseModel):

    correctness: int = Field(
        ge=0,
        le=5,
        description="生成答案与标准答案在事实和核心结论上的一致程度"
    )

    faithfulness: int = Field(
        ge=0,
        le=5,
        description="生成答案是否受到检索资料支持"
    )

    relevance: int = Field(
        ge=0,
        le=5,
        description="生成答案是否直接完整回答用户问题"
    )


# =========================================
# 2. 项目路径
# =========================================

basedir = pathlib.Path(
    __file__
).resolve().parent.parent

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


# =========================================
# 3. 加载评测数据
# =========================================

with open(
        dataset_path,
        'r',
        encoding='utf-8'
) as f:

    dataset = json.load(f)


# 第一次只跑1道题
# 确认代码没问题以后再删除这一行
dataset = dataset[:1]


# =========================================
# 4. 加载Embedding和向量数据库
# =========================================

model_name = 'BAAI/bge-m3'

embedding_model = get_embeddings(
    model_name
)

vectorstore = load_vectorstore(
    persist_directory,
    embedding_model
)


# =========================================
# 5. 加载Reranker
# =========================================

reranker = get_reranker()


# =========================================
# 6. 加载LLM
# =========================================

llm = create_llm(
    temperature=0
)


# =========================================
# 7. 创建Pydantic解析器
# =========================================

parser = PydanticOutputParser(
    pydantic_object=RAGEvaluation
)


# 自动生成告诉LLM应该输出什么格式的说明
format_instructions = (
    parser.get_format_instructions()
)


# =========================================
# 8. 保存评测分数
# =========================================

correctness_scores = []

faithfulness_scores = []

relevance_scores = []


# =========================================
# 9. 开始逐题评测
# =========================================

for item in dataset:

    question = item[
        'question'
    ]

    english_query = item[
        'english_query'
    ]

    reference_answer = item[
        'reference_answer'
    ]


    # =====================================
    # Retriever
    # 先召回20个候选Chunk
    # =====================================

    candidate_documents = search_chunk_bilingual(
        vectorstore,
        chinese_query=question,
        english_query=english_query,
        fetch_k=50,
        k=20
    )


    # =====================================
    # Reranker
    # 从20个候选Chunk中重新排序
    # 最后保留Top5
    # =====================================

    documents = rerank_documents(
        reranker,
        chinese_query=question,
        english_query=english_query,
        documents=candidate_documents,
        k=5
    )


    # =====================================
    # Generator
    # 根据Top5 Chunk生成答案
    # =====================================

    answer = generate_answer(
        llm=llm,
        query=question,
        documents=documents
    )


    # =====================================
    # 将Top5 Chunk拼接成Context
    # =====================================

    context = '\n\n'.join(
        doc.page_content
        for doc in documents
    )


    # =====================================
    # 构造Judge Prompt
    # =====================================

    judge_prompt = f"""
你是一个RAG系统评测器。

请评估下面生成的回答。

问题：
{question}

标准答案：
{reference_answer}

RAG检索资料：
{context}

RAG生成答案：
{answer}

请分别评价下面三个指标，每项0到5分。

correctness：
生成答案和标准答案在事实与核心结论上的一致程度。

faithfulness：
生成答案是否完全受到检索资料支持。
如果回答中存在检索资料没有提供的信息，应降低分数。

relevance：
生成答案是否直接、完整地回答用户问题。

请严格按照下面的格式输出：

{format_instructions}
"""


    # =====================================
    # LLM进行评测
    # =====================================

    response = llm.invoke(
        judge_prompt
    )


    # =====================================
    # Pydantic解析
    # str -> RAGEvaluation对象
    # =====================================

    result = parser.parse(
        response.content
    )


    # =====================================
    # 查看Pydantic对象
    # =====================================

    print(
        '\nPydantic结果：'
    )

    print(
        result
    )

    print(
        '\n结果类型：'
    )

    print(
        type(result)
    )


    # =====================================
    # 从Pydantic对象中取字段
    # =====================================

    correctness = (
        result.correctness
    )

    faithfulness = (
        result.faithfulness
    )

    relevance = (
        result.relevance
    )


    # =====================================
    # 保存分数
    # =====================================

    correctness_scores.append(
        correctness
    )

    faithfulness_scores.append(
        faithfulness
    )

    relevance_scores.append(
        relevance
    )


    # =====================================
    # 打印每道题结果
    # =====================================

    print(
        '\n======================'
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

    print(
        '\n标准答案：'
    )

    print(
        reference_answer
    )

    print(
        '\n评测结果：'
    )

    print(
        'Correctness:',
        correctness
    )

    print(
        'Faithfulness:',
        faithfulness
    )

    print(
        'Answer Relevance:',
        relevance
    )


# =========================================
# 10. 计算平均分
# =========================================

avg_correctness = (
    sum(correctness_scores)
    / len(correctness_scores)
)

avg_faithfulness = (
    sum(faithfulness_scores)
    / len(faithfulness_scores)
)

avg_relevance = (
    sum(relevance_scores)
    / len(relevance_scores)
)


# =========================================
# 11. 最终结果
# =========================================

print(
    '\n======================'
)

print(
    '最终RAG Answer Evaluation'
)

print(
    '问题数量：',
    len(dataset)
)

print(
    'Correctness平均分：',
    round(
        avg_correctness,
        2
    )
)

print(
    'Faithfulness平均分：',
    round(
        avg_faithfulness,
        2
    )
)

print(
    'Answer Relevance平均分：',
    round(
        avg_relevance,
        2
    )
)