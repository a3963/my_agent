from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from .llm_client import create_llm


# 加载.env
load_dotenv()
__all__ = ["create_llm", "generate_answer", "stream_generate_answer"]

# generate 与 stream 共用同一套 prompt 模板，保证一致
_PROMPT_TEMPLATE = """
你是一个专业知识助手。

请根据提供的资料回答问题。

要求：
1. 只能根据资料回答。
2. 如果资料没有答案，请明确说明不知道。


资料：
{context}


问题：
{question}


答案：
"""


def generate_answer(llm, query, documents):
    context = "\n\n".join(
        [
            doc.page_content
            for doc in documents
        ]
    )
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    return response.content


def stream_generate_answer(llm, question, documents):
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    chain = prompt | llm
    for chunk in chain.stream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content
