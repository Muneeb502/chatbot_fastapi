# import os
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# from dotenv import load_dotenv




# # LangChain imports
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_groq import ChatGroq




# # Load environment variables
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# # FastAPI app
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )




# # Request/Response models
# class ChatRequest(BaseModel):
#     message: str
#     history: Optional[List[Dict[str, str]]] = None


# class ChatResponse(BaseModel):
#     response: str


# # Load PDF and build vectorstore (cache in memory)
# class RAGChatbot:
#     def __init__(self, pdf_path: str):
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load_and_split()
#         embeddings = OpenAIEmbeddings()
#         self.vstore = FAISS.from_documents(docs, embeddings)
#         self.retriever = self.vstore.as_retriever(search_kwargs={"k": 4})
#         self.llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)



#     def format_history(self, history):
#         if not history:
#             return ""
#         lines = []
#         for msg in history[-4:]:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             lines.append(f"{role}: {msg['content']}")
#         return "\n".join(lines)


#     def get_context(self, query: str):
#         docs = self.retriever.get_relevant_documents(query)
#         return "\n\n".join(doc.page_content for doc in docs)


#     def build_prompt(self, query: str, context: str, history: str):
#         return f"""
# You are a helpful Personal AI assistant of Muneeb Use ONLY the following context to answer the user's question. If the answer is not in the context, say you don't know.

# Context:
# {context}

# Chat History:
# {history}

# User Question:
# {query}

# Assistant Response:"""

#     def chat(self, query: str, history: Optional[List[Dict[str, str]]]):
#         context = self.get_context(query)
#         hist = self.format_history(history)
#         prompt = self.build_prompt(query, context, hist)
#         result = self.llm.invoke(prompt)
#         if hasattr(result, "content"):
#             return result.content
#         return str(result)


# # Instantiate chatbot (singleton)
# chatbot = RAGChatbot("data.pdf")



# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     try:
#         answer = chatbot.chat(request.message, request.history)
#     except Exception as e:
#         answer = f"Sorry, an error occurred: {e}"
#     return ChatResponse(response=answer) 


import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
import uvicorn



# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str

# Load PDF and build vectorstore (cache in memory)
class RAGChatbot:
    def __init__(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        embeddings = OpenAIEmbeddings()
        self.vstore = FAISS.from_documents(docs, embeddings)
        self.retriever = self.vstore.as_retriever(search_kwargs={"k": 4})
        self.llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)

    def format_history(self, history):
        if not history:
            return ""
        lines = []
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_context(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def build_prompt(self, query: str, context: str, history: str):
        return f"""
You are a helpful AI assistant for Indus Medical. Use ONLY the following context to answer the user's question. If the answer is not in the context, say you don't know.

Context:
{context}

Chat History:
{history}

User Question:
{query}

Assistant Response:"""

    def chat(self, query: str, history: Optional[List[Dict[str, str]]]):
        context = self.get_context(query)
        hist = self.format_history(history)
        prompt = self.build_prompt(query, context, hist)
        result = self.llm.invoke(prompt)
        if hasattr(result, "content"):
            return result.content
        return str(result)

# Instantiate chatbot (singleton)
chatbot = RAGChatbot("data.pdf")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        answer = chatbot.chat(request.message, request.history)
    except Exception as e:
        answer = f"Sorry, an error occurred: {e}"
    return ChatResponse(response=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
