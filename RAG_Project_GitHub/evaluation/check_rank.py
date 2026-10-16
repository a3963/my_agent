from pathlib import Path

from backend.rag.embedding import get_embeddings
from backend.rag.vectorstore import load_vectorstore


base_dir = Path(__file__).resolve().parent.parent

persist_directory = base_dir / 'data' / 'chroma_db_m3'

embedding_model = get_embeddings(
    "BAAI/bge-m3"
)

vectorstore = load_vectorstore(
    persist_directory,
    embedding_model
)


question = "超快激光焊接透明材料时，热积累机制是如何导致材料熔化并形成焊接区域的？"

correct_source = "Jia 等 - 2025 - Ultrafast laser welding of transparent materials from principles to applications.pdf"


documents = vectorstore.similarity_search(
    question,
    k=100
)


for index,doc in enumerate(documents,start=1):

    source = doc.metadata.get(
        'source',
        ''
    )

    if source == correct_source:

        print("正确论文第一次出现排名：",index)

        print(
            doc.page_content[:500]
        )

        break

else:

    print("Top100都没有找到正确论文")