from typing import List
from langchain_chroma import Chroma
from langchain.schema import Document


def index_docs(texts: List[str], vector_store: Chroma) -> None: 
    vector_store.add_texts(texts)

def retrieve_docs(query: str, vector_store: Chroma) -> List[Document]:
    return vector_store.similarity_search(query)

