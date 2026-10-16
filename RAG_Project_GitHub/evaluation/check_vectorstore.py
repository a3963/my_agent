from collections import Counter
from pathlib import Path

from backend.rag.embedding import get_embeddings
from backend.rag.vectorstore import load_vectorstore


base_dir = Path(__file__).resolve().parent.parent

persist_directory = base_dir / 'data' / 'chroma_db'

embedding_model = get_embeddings(
    "BAAI/bge-small-zh-v1.5"
)

vectorstore = load_vectorstore(
    persist_directory,
    embedding_model
)


data = vectorstore.get()

metadatas = data['metadatas']

sources = []

for metadata in metadatas:
    source = metadata.get('source','')
    sources.append(source)


source_count = Counter(sources)

print('Chroma总chunk数：',len(sources))

print('\n每篇论文chunk数量：')

for source,count in source_count.most_common():
    print(count,source)