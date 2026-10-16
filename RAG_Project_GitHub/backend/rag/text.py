try:
    from rag.llm_client import create_llm
except ImportError:
    from llm_client import create_llm


llm = create_llm()


result = llm.invoke("你好")


print(result.content)
