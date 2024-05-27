# Building Your Own Retrieval Augmented Generation (RAG) Application Using Langchain

Retrieval Augmented Generation (RAG) is a powerful technique for enhancing the capabilities of Language Learning Models (LLMs) by providing them with additional data. This is particularly useful for reasoning about private or new data beyond the LLM's training cut-off. In this blog, we'll guide you through building a RAG application using Langchain.

## What is RAG?

RAG stands for Retrieval Augmented Generation. It combines the reasoning ability of LLMs with the specific data needed to answer a query accurately. LLMs, while powerful, have a knowledge cut-off date and cannot reason about data introduced after this point or private data. RAG addresses this by retrieving relevant information at runtime and feeding it into the LLM.

![RAG Workflow](https://truera.com/wp-content/uploads/2023/09/truera-architecture-for-chatot-figure-1-1024x561.png)

## Setting Up the Python Environment

To create a RAG application, you'll need the following:

- **Python version**: 3.8 - 3.12 (Virtual Environment optional)
- **OpenAI API Key**
- **Required Libraries**: `langchain`, `langchain-community`, `langchain-openai`, `FAISS-cpu`, `tiktoken`

First, install the necessary libraries:

```bash
!pip install -U langchain-community faiss-cpu langchain-openai tiktoken
```

Set your OpenAI API key:

```python
import os

os.environ["OPENAI_API_KEY"] = "enter api key here"
```

## Components of a RAG Application

A typical RAG application has two main components:

1. **Indexing**: Ingesting and indexing data, usually done offline.
2. **Retrieval and Generation**: Retrieving relevant data and generating responses at runtime.

### Indexing

#### Load Data

The first step is to load your data using DocumentLoaders.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("/content/state_of_the_union.txt")
documents = loader.load()
```

#### Split Data

Split large documents into smaller chunks for efficient searching and model processing.

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
```

#### Store Data

Store and index the split documents using a VectorStore and Embeddings model.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
```

### Retrieval and Generation

#### Retrieve Data

Retrieve relevant document chunks based on the user's query.

```python
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
```

You can also convert the VectorStore into a Retriever for use in other LangChain methods.

```python
retriever = db.as_retriever()
docs = retriever.invoke(query)
```

#### Save and Load Index

Save the VectorStore locally and reload it as needed.

```python
db.save_local("db_name")

new_db = FAISS.load_local("db_name", embeddings, allow_dangerous_deserialization=True)
docs = new_db.similarity_search(query)
```

#### Initialize LLM

Initialize a ChatModel for generating responses.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

#### Create a RAG Chain

Create a RAG chain by combining the retriever with a prompt template and LLM.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
```

#### Usage

Invoke the RAG chain with a user query.

```python
rag_chain.invoke("What the president say about Ketanji Brown Jackson?")
```

