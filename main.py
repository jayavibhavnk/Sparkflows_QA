##streamlit code 

import streamlit as st
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

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
    system_message = SystemMessagePromptTemplate.from_template(
    """
    You are a chatbot tasked with responding to questions about the documentation of the LangChain library and project.

    You should never answer a question with a question, and you should always respond with the most relevant documentation page.

    Do not answer questions that are not about the LangChain library or project.

    Given a question, you should respond with the most relevant documentation page by following the relevant context below:\n
    {context}
    """
        )
    human_message = HumanMessagePromptTemplate.from_template("{question}")

    return system_message, human_message

def query_from_doc1(text):
    global chat_history
    qa = ChatVectorDBChain.from_llm(llm, db)
    prompt_template = "You are a helpful assisstant"
    result = qa({"question": get_prompt() + text, "chat_history": st.session_state.chat_history})
    # print(result["answer"])
    st.session_state.chat_history = [(text, result["answer"])]
    return result["answer"]

def query_from_doc(text):
    llm = ChatOpenAI(model="gpt-4") 

    system_message, human_message = get_prompt()
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )

    ans = conversation_chain(text)

    return ans['answer']

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

def main():

    load_faiss_embeddings("db_faiss")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

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


if __name__ == "__main__":
    main()
