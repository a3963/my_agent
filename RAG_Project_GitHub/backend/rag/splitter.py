from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 20
    )
    chunks = splitter.split_documents(documents)
    return chunks