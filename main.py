##streamlit code 

import streamlit as st
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = st.secrets.OPENAI_API_KEY

st.title("Sparkflows Documentation")
st.subheader(
    "Ask anything about the sparkflows documentation",
)

db = None

st.session_state.chat_history = []

llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        model_name='gpt-4-0125-preview',
        temperature=0.8
    )

def load_faiss_embeddings(path):
    global db
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
        )
    db = FAISS.load_local('db_faiss', embeddings)

    print("db loaded")

def query_faiss(query):
    global db
    ans = db.similarity_search(query)
    print(ans)
    # return ans.page_content

def get_prompt():
    prompt_template = """
    You are an expert support agent at Sparkflows

        Your task is to answer customer queries related to sparkflows. If you don't know any answer, don't try to make up an answer. Just say that you don't know and to contact the company support.

        Don't be overconfident and don't hallucinate. Ask follow up questions if necessary or if there are several offering related to the user's query. Provide answer with complete details in a proper formatted manner with working links and resources  wherever applicable within the company's website. Never provide wrong links.

        One of the major queries from the user will be: how do i install sparkflows 
        since sparkflows can be installed on several platforms you will give a generalised answer and then prompt the user and ask them what platform do they want to run the product on 
        when a specific platform is given, you will provide a detailed instruction on how to install it
    """

def query_from_doc(text):
    global chat_history
    qa = ChatVectorDBChain.from_llm(llm, db)
    prompt_template = "You are a helpful assisstant"
    result = qa({"question":prompt_template+ text, "chat_history": st.session_state.chat_history})
    # print(result["answer"])
    st.session_state.chat_history = [(text, result["answer"])]
    return result["answer"]

def query_with_link(query):
    new_db = db.similarity_search(query)
    relevant_links = [i.metadata['source'] for i in new_db]
    rel_links = []
    for i in relevant_links:
        if i not in rel_links:
            rel_links.append(i + "\n")
    links = '\n'.join(rel_links)
    response_from_chatgpt = query_from_doc(query)
    final_response = response_from_chatgpt + "\n\nHere are some of the relevant links: \n \n" +links

    return final_response

if 1:

    load_faiss_embeddings("db_faiss")

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about Sparkflows"}]

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                msg = query_with_link(prompt)
                st.write(msg)
                message = {"role": "assistant", "content": msg}
                st.session_state.messages.append(message) 
