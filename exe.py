# Install required packages 
import os
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredExcelLoader
from dotenv import load_dotenv



#  Load API Key 


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY') 
# Step 1 for loading the data 

excel_path = "cs401.xlsx"  
loader = UnstructuredExcelLoader(excel_path)
document = loader.load()

# Step 2: Split the Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=110, chunk_overlap=50)
split_document = text_splitter.split_documents(document)

# Step 3: Generate Embeddings with HuggingFace Sentence Transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Step 4 Store embeddings in vector database
vectorstore = FAISS.from_documents(documents=split_document, embedding=embeddings)

# Step 5 creating retriever part 
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 6: Initialize  LLM 
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)


# Step 7 Create  RAG Chain
system_prompt = (
    "You are a helpful assistant. Answer the questions using only the information provided in the context. "
    "If the answer is not in the context, politely tell the user that the answer is not available and suggest they rephrase or ask a different question."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Step 8 Ask  Question
query = "Tell me what are the basic rights for citizens?"  
response = rag_chain.invoke({"input": query})
print("Query:", query)
print(response['answer'])
