from .rag.retriever import search_document
from .rag.generator import generate_answer,stream_generate_answer

class RAGChain:
    def __init__(self,vectorstore,llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def ask(self,question):
        documents = search_document(self.vectorstore,question)
        answer = generate_answer(llm=self.llm,query=question,documents=documents)
        return answer

    def stream(self,question):
        documents = search_document(self.vectorstore,question)
        for chunk in stream_generate_answer(
                self.llm,
                question,
                documents
        ):
            yield chunk


