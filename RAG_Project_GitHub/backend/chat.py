from pathlib import Path
from .rag.embedding import get_embeddings
from .rag.vectorstore import load_vectorstore
from .rag.generator import create_llm
from .rag_chain import RAGChain


base_dir = Path(__file__).resolve().parent.parent
# 1. 加载 Embedding
embedding_model = get_embeddings(
    "BAAI/bge-small-zh-v1.5"
)
# 2. 加载 Chroma
persist_directory = (
    base_dir
    / "data"
    / "chroma_db"
)
vectorstore = load_vectorstore(
    persist_directory,
    embedding_model
)
# 3. 创建 LLM
llm = create_llm()
# 4. 创建完整 RAG 对象
rag = RAGChain(
    vectorstore,
    llm
)
# 5. 用户问题
question = "什么是激光透射焊接？"
# 6. 流式回答
for chunk in rag.stream(question):
    print(chunk,end="",flush=True)