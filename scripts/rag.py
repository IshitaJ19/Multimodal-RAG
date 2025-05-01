from typing import List
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


template = """
You are an assistant that is tasked with answering questions.
Use the following pieces of context to generate an answer.
If you don't know the answer, just say that you don't know.
Use at most three sentences and keep the answer concise.
Question: {question}
Context: {context}
Answer: 
"""

def answer_question(question: str, documents: List[Document], model: OllamaLLM) -> str:

    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

