import os
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

loader = WebBaseLoader(
web_paths=("https://kbourne.github.io/chapter1.html",),
bs_kwargs=dict(
parse_only=bs4.SoupStrainer(
class_=("post-content", "post-title",
"post-header"))),)
docs = loader.load()
# print(docs)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = SemanticChunker(embedder)
splits = text_splitter.split_documents(docs)
# print(splits)

# For Embedding and indexing the chunks
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedder)
retriever = vectorstore.as_retriever()

prompt = hub.pull("jclemens24/rag-prompt")
print(prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

rag_chain = (
{"context": retriever | format_docs,
"question": RunnablePassthrough()}
| prompt
| model
| StrOutputParser()
)

result = rag_chain.invoke("What are the advantages of using RAG?")
print(result)


















