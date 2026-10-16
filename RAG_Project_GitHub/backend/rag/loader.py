#from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain_core.documents import Document
def load_pdf(filepath):
    reader = PdfReader(str(filepath))
    documents = []
    for page_index,page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.encode('utf-16','surrogatepass').decode('utf-16','ignore')

        if text.strip():
            doc = Document(page_content = text,
                           metadata = {'source':filepath.name,
                                       '页数':page_index+1})
            documents.append(doc)
    return documents
# def load_pdf(filepath:str):
#     loader = PyPDFLoader(filepath)
#     document = loader.load()
#     return document