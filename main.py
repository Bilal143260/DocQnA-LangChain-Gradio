import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = #add your openai key here

def process_pdf(file):
    loader = PyPDFLoader(file.name)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(all_splits, OpenAIEmbeddings())
    
    return vectorstore

def query_chain(vectorstore, question):
    template = """Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                    Use three sentences maximum and keep the answer as concise as possible. 
                    {context}
                    Question: {question}
                    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": question})
    return result["result"]

def app(file, question):
    vectorstore = process_pdf(file)
    answer = query_chain(vectorstore, question)
    return answer

iface = gr.Interface(
    fn=app,
    inputs=[
        gr.inputs.File(label="Upload PDF"),
        gr.inputs.Textbox(label="Your question")
    ],
    outputs=gr.outputs.Textbox(label="Answer"),
    title="PDF READER AND Q/A BOT"
)

iface.launch()
