import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings  
from langchain_core.prompts import (PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import streamlit as st
from dotenv import load_dotenv

load_dotenv("api_key.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

st.header("Eklavya's Cheating ChatBot", divider=True)

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
word_limit = st.slider("Set Word Limit for the Answer", min_value=50, max_value=1000, value=300, step=50)

if uploaded_file:
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    
    pdf_path = f"./temp/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader_pdf = PyPDFLoader(pdf_path)
    docs_list = loader_pdf.load()

    token_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=300, chunk_overlap=50)
    docs_list_token_split = token_splitter.split_documents(docs_list)

    model = SentenceTransformer("all-MiniLM-L6-v2")  
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    documents = [Document(page_content=doc.page_content) for doc in docs_list_token_split]

    vectorstore = Chroma.from_documents(
        documents,
        embedding=embedding_function,
        persist_directory="./temp/vectorstore"
    )

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'lambda_mult': 0.7}
    )

    PROMPT_RETRIEVING_S = '''You will receive a question from a user. Answer the question using only the provided context.'''
    PROMPT_TEMPLATE_RETRIEVING_H = f'''This is the question:
    {{question}}

    This is the Context:
    {{context}}

    Provide an answer with a maximum of {word_limit} words.'''

    prompt_retrieving_s = SystemMessage(PROMPT_RETRIEVING_S)
    prompt_template_retrieving_h = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE_RETRIEVING_H)
    chat_prompt_retrieving = ChatPromptTemplate([prompt_retrieving_s, prompt_template_retrieving_h])

    chat = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

    str_output_parser = RunnablePassthrough()
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | chat_prompt_retrieving
        | chat
        | str_output_parser
    )

    question = st.text_input("\n Type Your Question Here :")

    if st.button("Ask"):
        if question:
            response_placeholder = st.empty()
            response_text = ""
            
            result = chain.stream(question)
            
            for chunk in result:
                response_text += chunk.content
                response_placeholder.markdown(response_text)
        else:
            st.warning("Please enter a question to proceed.", icon = "warning")
