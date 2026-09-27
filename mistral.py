# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
device = "cuda"
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import json
import streamlit as st

import markdown

import time
from datetime import datetime

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import Chroma, FAISS
import nest_asyncio

from langchain.schema.runnable import RunnablePassthrough

import textwrap

from playwright.async_api import async_playwright

# from transformers import pipeline

import asyncio

import os

import torch

import requests

from langchain_together import ChatTogether

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_webpage_size(url):
    response = requests.get(url)
    return len(response.content)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, sys_prompt):
    system_prompt = B_SYS + sys_prompt + E_SYS
    template = B_INST + system_prompt +  instruction + E_INST
    return template



def load_tokenizer_and_llm():

    llm = ChatTogether(
        model="openai/gpt-oss-120b",
        max_tokens = 2048,
        temperature=0.1,
        together_api_key=os.getenv("TOGETHER_API_KEY")
        #together_api_key = os.getenv("env")
    )

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", quantization_config=quantization_config)
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    # llm_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     use_cache=True,
    #     device_map = "auto",
    #     max_new_tokens = 2048,
    #     do_sample=True,
    #     top_k=7,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.eos_token_id
    # )

    # llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm

instruction = "Given the context that has been provided. \n {context}, Answer the following question: \n{question}"

sys_prompt = """You are a medical diagnosis expert.
You will be given medical context to answer from. Answer the questions with as much detail as possible. Only answer medical questions, nothing else
In case you do not know the answer, you can say "I don't know" or "I don't understand".
In all other cases provide an answer to the best of your ability. If someone asks about treatment options, link https://clinicaltrials.gov/ at the end"""


prompt_sys = get_prompt(instruction, sys_prompt)


template = PromptTemplate(template=prompt_sys, input_variables=['context', 'question'])


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text

def process_llm_response(llm_response):
    raw_text = llm_response['text']

    response_html = markdown.markdown(
        raw_text,
        extensions=['tables', 'fenced_code']
    )

    sources_list = [
        source.metadata['source']
        for source in llm_response['context']
    ]

    return {
        "answer": response_html,
        "sources": sources_list
    }





def load_data():

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={
            'device': torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        },
        encode_kwargs={'normalize_embeddings': True}
    )

    index_path = "faiss_index"

    # ---------------------------------------------------------
    # LOAD EXISTING INDEX
    # ---------------------------------------------------------

    if os.path.exists(index_path):
        log("Saved FAISS index found. Loading from disk...")

        start = time.perf_counter()

        db = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        log(
            f"FAISS index loaded in "
            f"{time.perf_counter() - start:.2f} sec"
        )

        return db


    # ---------------------------------------------------------
    # OTHERWISE BUILD IT
    # ---------------------------------------------------------

    log("No saved FAISS index found. Building new index...")

    articles = [
        "https://www.cancer.gov/resources-for/patients",
        "https://www.cancer.org/cancer/types.html,",
        "https://www.cancer.org/cancer/diagnosis-staging/staging.html",
        "https://www.cancer.gov/about-cancer"
    ]

    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    base_url = "https://www.cancer.gov/publications/dictionaries/cancer-terms"
    new_base_url = "https://www.cancer.gov/publications/dictionaries/cancer-drug"

    for letter in alphabets:
        articles.append(f"{base_url}/expand/{letter}")
        articles.append(f"{new_base_url}/expand/{letter}")

    loader = AsyncChromiumLoader(articles)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    chunked_documents = text_splitter.split_documents(
        docs_transformed
    )

    log("Embedding chunks and building FAISS...")

    db = FAISS.from_documents(
        chunked_documents,
        embeddings
    )

    # ---------------------------------------------------------
    # SAVE INDEX
    # ---------------------------------------------------------

    log("Saving FAISS index to disk...")

    db.save_local(index_path)

    log("FAISS index saved.")

    return db


query = "What are the treatment options for a patient with colon adenocarcinoma stage 2 carrying mutations in TP53, FBXW7, APC, as well as CDK6 amplification and EGFR amplification?" 

def process_query(query, llm, db):

    total_start = time.perf_counter()

    log("Starting process_query()")
    log(f"Query: {query}")

    # ---------------------------------------------------------
    # CREATE RETRIEVER
    # ---------------------------------------------------------

    retriever_start = time.perf_counter()

    retriever = db.as_retriever()

    log(
        f"Retriever created in "
        f"{time.perf_counter() - retriever_start:.4f} sec"
    )


    # ---------------------------------------------------------
    # BUILD LLM CHAIN
    # ---------------------------------------------------------

    chain_start = time.perf_counter()

    llm_chain = LLMChain(
        llm=llm,
        prompt=template
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | llm_chain
    )

    log(
        f"RAG chain constructed in "
        f"{time.perf_counter() - chain_start:.4f} sec"
    )


    # ---------------------------------------------------------
    # RETRIEVAL + LLM GENERATION
    # ---------------------------------------------------------

    log("Invoking RAG chain...")
    invoke_start = time.perf_counter()

    ans = rag_chain.invoke(query)

    invoke_time = time.perf_counter() - invoke_start

    log(
        f"Retrieval + LLM invocation finished in "
        f"{invoke_time:.2f} sec"
    )


    # ---------------------------------------------------------
    # RESPONSE FORMATTING
    # ---------------------------------------------------------

    processing_start = time.perf_counter()

    response = process_llm_response(ans)

    log(
        f"Response processing finished in "
        f"{time.perf_counter() - processing_start:.4f} sec"
    )

    log(
        f"process_query() COMPLETE in "
        f"{time.perf_counter() - total_start:.2f} sec"
    )

    return response

