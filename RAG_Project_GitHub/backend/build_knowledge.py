from rag.loader import load_pdf
from rag.splitter import split_document
from rag.embedding import get_embeddings
from rag.vectorstore import create_vectorstore
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
print(base_dir)
filepath = base_dir/'data'/'documents'
all_documents = []
for document in filepath.glob('*.pdf'):
    print('正在读取:',document.name)
    documents = load_pdf(document)
    all_documents.extend(documents)
print('打印完成')
chunks = split_document(all_documents)
print('chunk一共：',len(chunks))
print(type(chunks[0]))
print(chunks[0])
model_name="BAAI/bge-m3"
embedding_model = get_embeddings(model_name)
persist_directory = base_dir/'data'/'chroma_db_m3'
vectorstore = create_vectorstore(chunks,embedding_model,persist_directory)
print('知识库构建完成')