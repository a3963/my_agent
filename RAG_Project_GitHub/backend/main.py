import json
import shutil
import hashlib
from typing import Annotated
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from .rag.embedding import get_embeddings
from .rag.generator import create_llm
from .rag.vectorstore import load_vectorstore, add_documents
from .rag.loader import load_pdf
from .rag.splitter import split_document
from .rag_chain import RAGChain


@asynccontextmanager
async def lifespan(app:FastAPI):
    print('RAG程序开始启动')

    pathfile = Path(__file__).resolve().parent.parent
    persist_directory = pathfile / 'data' / 'chroma_db'

    model_name = "BAAI/bge-small-zh-v1.5"
    embedding_name = get_embeddings(model_name)

    vectorstore = load_vectorstore(
        persist_directory,
        embedding_name
    )

    llm = create_llm()

    app.state.rag = RAGChain(
        vectorstore,
        llm
    )

    print('RAG初始化成功')

    yield

    print('FASTAPI进程结束')


def get_file_hash(file_bytes:bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def load_hash_records(record_path:Path):
    if record_path.exists():
        with open(record_path,'r',encoding='utf-8') as f:
            return json.load(f)

    return {}


def save_hash_records(record_path:Path,records):
    with open(record_path,'w',encoding='utf-8') as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    question:str


@app.get("/")
def root():
    return {
        "message":"RAG API is running"
    }


@app.get("/upload-page",response_class=HTMLResponse)
def upload_page():
    return """
    <html>
    <body>

    <h2>RAG知识库文件上传</h2>

    <form action="/upload"
          enctype="multipart/form-data"
          method="post">

        <input name="files"
               type="file"
               multiple>

        <br><br>

        <button type="submit">
            上传并入库
        </button>

    </form>

    </body>
    </html>
    """


@app.post("/ask")
def ask(payload:AskRequest,request:Request):

    rag = request.app.state.rag

    answer = rag.ask(
        payload.question
    )

    return {
        "question":payload.question,
        "answer":answer
    }


@app.post("/stream")
def stream(payload:AskRequest,request:Request):

    rag = request.app.state.rag

    return StreamingResponse(
        rag.stream(payload.question),
        media_type="text/plain; charset=utf-8"
    )


@app.post("/upload")
def upload_files(
        files:Annotated[list[UploadFile],File()],
        request:Request
):
    base_dir = Path(__file__).resolve().parent.parent

    upload_dir = base_dir / 'data' / 'documents'
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    record_path = base_dir / 'data' / 'file_hash_records.json'

    hash_records = load_hash_records(record_path)

    existing_hashes = set(
        hash_records.values()
    )

    rag = request.app.state.rag

    saved_files = []
    skipped_files = []

    for file in files:

        file_bytes = file.file.read()

        file_hash = get_file_hash(
            file_bytes
        )

        if file_hash in existing_hashes:
            skipped_files.append(
                file.filename
            )
            continue

        file_path = upload_dir / file.filename

        with open(file_path,'wb') as buffer:
            buffer.write(file_bytes)

        documents = load_pdf(
            file_path
        )

        chunks = split_document(
            documents
        )

        add_documents(
            rag.vectorstore,
            chunks
        )

        hash_records[file.filename] = file_hash

        existing_hashes.add(
            file_hash
        )

        saved_files.append(
            file.filename
        )

    save_hash_records(
        record_path,
        hash_records
    )

    return {
        "message":"文件处理完成",
        "saved_files":saved_files,
        "skipped_files":skipped_files
    }