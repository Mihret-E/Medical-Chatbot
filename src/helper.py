from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
#from langchain_community.schema import Document
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings




#Extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents= loader.load()
    return documents


#Filtering only the necessary information

def filter_to_minimal_docs(docs:List[Document]) ->List[Document]:

    Minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        Minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return Minimal_docs


# Split the documnet into smaller chunks

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap =20,
    )
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk

#Next step is embedding model  hugging face embedding model will be downloaded here

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings

embedding = download_embeddings()