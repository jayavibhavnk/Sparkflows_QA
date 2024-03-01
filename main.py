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

def load_faiss_embeddings(path):
    global db
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.secrets.OPENAI_API_KEY
        )
    db = FAISS.load_local('db_faiss', embeddings)    

    st.session_state.vector_store = db

    print("db loaded")

def query_faiss(query):
    global db
    ans = db.similarity_search(query)
    print(ans)
    # return ans.page_content

def get_prompt():
    system_message = SystemMessagePromptTemplate.from_template(
    """
    You are a customer interaction agent for Sparkflows.io, 

        Do not answer questions that are not about Sparkflows.

        Please do not make up links, do not return urls or links to any documentation

    if a customer asks how sparkflows can be installed, you will give the user all options to install it and further prompt them to select which platform they would like to install it on
    if the customer shares which platform they want to install sparkflows on, you will give a detailed explanation on that

    Don't be overconfident and don't hallucinate. Ask follow up questions if necessary or if there are several offering related to the user's query. Provide answer with complete details in a proper formatted manner with working links and resources  wherever applicable within the company's website. Never provide wrong links.

    try to keep the conversation engaging


    
    Given a question, you should respond with the most relevant documentation page by following the relevant context below and also return relevant links from this:\n
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

def query_from_doc2(text):
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

    st.session_state.chat_history = ans["chat_history"]
    
    return ans['chat_history']

def query_from_doc(text):
    response = st.session_state.conversation({"question": text, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history = response['chat_history']
    
    return response['answer']

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

def get_conversation_chain(vector_store:FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model="gpt-4")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
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
    return conversation_chain

def main():

    OPENAI_API_KEY = st.secrets.OPENAI_API_KEY
    
    db = None
    
    llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            model_name='gpt-4-0125-preview',
            temperature=0.8
        )
    
    load_faiss_embeddings("db_faiss")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    st.title("Sparkflows Documentation")
    st.subheader(
        "Ask anything about the sparkflows documentation",
    )
    
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

    system_message_prompt, human_message_prompt = get_prompt()
    
    st.session_state.conversation = get_conversation_chain(st.session_state.vector_store, system_message_prompt, human_message_prompt)

if __name__ == "__main__":
    main()
