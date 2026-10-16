from langchain_huggingface import HuggingFaceEmbeddings



def get_embeddings(model_name):
    embedding_model = HuggingFaceEmbeddings(model_name = model_name)
    return embedding_model