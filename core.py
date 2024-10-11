import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangchain

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def run_llm(query:str, chat_history:List[Dict[str, Any]] = [] ) ->Any:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeLangchain.from_existing_index(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)

    # Set up retrieval-based QA system
    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(),
        prompt = rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever= history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input":query, "chat_history":chat_history})

    new_result = {
        "query": query,
        "result": result["answer"],
        "source": result["context"]
    }

    return new_result