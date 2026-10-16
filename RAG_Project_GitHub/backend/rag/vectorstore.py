from langchain_chroma import Chroma

def create_vectorstore(chunks,embedding_model,persist_directory):
    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=embedding_model,
                                        persist_directory = persist_directory)
    return vectorstore

def load_vectorstore(persist_directory,embedding_model):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vectorstore

def add_documents(vectorstore,chunks):
    vectorstore.add_documents(chunks)
    return vectorstore