"""
Example script that builds a BGE3-powered RAG pipeline using SiliconFlow's DeepSeek Chat
endpoint for generation and Pinecone or Chroma for vector storage.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

BGE3_DIMENSION = 1024
DEFAULT_PROMPT = """Answer the question based only on the following context:
{context}

Question: {question}
"""


def load_documents(file_path: Path) -> List[Document]:
    """Load PDF documents using the settings from full_basic_rag.ipynb."""
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def split_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(list(documents))


def get_bge3_embeddings(api_key: str, base_url: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=api_key,
        base_url=base_url,
    )


def ensure_pinecone_index(client: Pinecone, index_name: str) -> None:
    existing = {idx["name"] for idx in client.list_indexes()}
    if index_name not in existing:
        logger.info("Creating Pinecone index %s", index_name)
        client.create_index(
            name=index_name,
            dimension=BGE3_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def build_vectorstore(
    splits: List[Document],
    embeddings: OpenAIEmbeddings,
    store: str,
    index_name: str,
    persist_dir: Optional[Path],
    pinecone_api_key: Optional[str],
    pinecone_api_host: Optional[str],
    top_k: int,
) -> Runnable:
    if store == "pinecone":
        if not pinecone_api_key or not pinecone_api_host:
            raise ValueError("Pinecone API configuration is required for Pinecone mode.")
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        os.environ["PINECONE_API_HOST"] = pinecone_api_host

        client = Pinecone(api_key=pinecone_api_key)
        ensure_pinecone_index(client, index_name)
        vectorstore = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name,
            namespace=None,
        )
    else:
        persist_directory = str(persist_dir or Path(".chroma"))
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=index_name,
            persist_directory=persist_directory,
        )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def build_llm(api_key: str, base_url: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
    )


def build_rag_chain(retriever: Runnable, llm: ChatOpenAI, prompt_template: str) -> Runnable:
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SiliconFlow DeepSeek RAG example")
    parser.add_argument("--file", required=True, help="Path to the PDF to index")
    parser.add_argument("--index-name", required=True, help="Vector store index or collection name")
    parser.add_argument("--vector-store", choices=["pinecone", "chroma"], default="chroma")
    parser.add_argument("--persist-dir", help="Persistence directory for Chroma collections")
    parser.add_argument("--top-k", type=int, default=4, help="Number of documents to retrieve")
    parser.add_argument("--question", help="Question to ask via CLI (omit to start FastAPI)")
    parser.add_argument("--host", default="0.0.0.0", help="FastAPI host when serving")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port when serving")
    parser.add_argument("--siliconflow-api-key", help="SiliconFlow API key")
    parser.add_argument(
        "--siliconflow-base-url",
        default="https://api.siliconflow.cn/v1",
        help="Base URL for SiliconFlow (override for testing)",
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-ai/DeepSeek-V3",
        help="DeepSeek chat model name hosted on SiliconFlow",
    )
    parser.add_argument("--pinecone-api-key", help="Pinecone API key")
    parser.add_argument("--pinecone-api-host", help="Pinecone API host")
    return parser.parse_args()


def create_app(rag_chain: Runnable) -> FastAPI:
    app = FastAPI(title="SiliconFlow DeepSeek RAG")

    class QueryRequest(BaseModel):
        question: str

    @app.post("/query")
    async def query(request: QueryRequest):
        logger.info("Received API query: %s", request.question)
        try:
            answer = rag_chain.invoke(request.question)
            return {"answer": answer}
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("Failed to process query")
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    load_dotenv()
    args = parse_args()

    siliconflow_api_key = args.siliconflow_api_key or os.getenv("SILICONFLOW_API_KEY")
    siliconflow_base_url = args.siliconflow_base_url
    pinecone_api_key = args.pinecone_api_key or os.getenv("PINECONE_API_KEY")
    pinecone_api_host = args.pinecone_api_host or os.getenv("PINECONE_API_HOST")

    if not siliconflow_api_key:
        raise ValueError("SILICONFLOW_API_KEY is required for embeddings and chat.")

    pdf_path = Path(args.file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Could not find input file: {pdf_path}")

    logger.info("Loading documents from %s", pdf_path)
    docs = load_documents(pdf_path)
    logger.info("Splitting documents (chunks of 1000 with 200 overlap)")
    splits = split_documents(docs)

    logger.info("Preparing BGE3 embeddings via SiliconFlow")
    embeddings = get_bge3_embeddings(siliconflow_api_key, siliconflow_base_url)
    retriever = build_vectorstore(
        splits=splits,
        embeddings=embeddings,
        store=args.vector_store,
        index_name=args.index_name,
        persist_dir=Path(args.persist_dir) if args.persist_dir else None,
        pinecone_api_key=pinecone_api_key,
        pinecone_api_host=pinecone_api_host,
        top_k=args.top_k,
    ).with_config(run_name="retriever", tags=["retriever", args.vector_store])

    llm = build_llm(
        api_key=siliconflow_api_key,
        base_url=siliconflow_base_url,
        model=args.deepseek_model,
    ).with_config(run_name="deepseek-chat", tags=["deepseek", "siliconflow"])

    rag_chain = build_rag_chain(retriever, llm, DEFAULT_PROMPT)

    if args.question:
        logger.info("Submitting CLI query: %s", args.question)
        answer = rag_chain.invoke(args.question)
        print("Answer:\n" + answer)
    else:
        logger.info("Launching FastAPI app at http://%s:%s/query", args.host, args.port)
        app = create_app(rag_chain)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
