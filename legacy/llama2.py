from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
device = "cuda"
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import json
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import Chroma, FAISS
import nest_asyncio

from langchain.schema.runnable import RunnablePassthrough

import textwrap

from playwright.async_api import async_playwright

from transformers import pipeline

import asyncio

import os

from dotenv import load_dotenv
load_dotenv()



import requests

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")



def get_webpage_size(url):
    response = requests.get(url)
    return len(response.content)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, sys_prompt):
    system_prompt = B_SYS + sys_prompt + E_SYS
    template = B_INST + system_prompt +  instruction + E_INST
    return template

from langchain_together import ChatTogether


print(torch.cuda.is_available())

def load_tokenizer_and_llm_llama2():



    llm = ChatTogether(
        model = "togethercomputer/llama-2-7b-chat",
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

    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", quantization_config=quantization_config, token = os.getenv('HF_KEY'))
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = os.getenv('HF_KEY'))

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

from langchain_together.embeddings import TogetherEmbeddings


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text

def process_llm_response(llm_response):
    response_text = wrap_text_preserve_newlines(llm_response)
    # Extracting sources into a list
    sources_list = [source.metadata['source'] for source in llm_response['source_documents']]

    # Returning a dictionary with separate keys for text and sources
    return {"answer": response_text, "sources": sources_list}




def load_data_llama2():
    # nest_asyncio.apply()

    articles = ["https://www.cancer.gov/resources-for/patients",
                "https://www.cancer.org/cancer/types.html,",
                "https://www.cancer.org/cancer/diagnosis-staging/staging.html",
                "https://www.cancer.gov/about-cancer"]

    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    base_url = "https://www.cancer.gov/publications/dictionaries/cancer-terms"
    new_base_url = "https://www.cancer.gov/publications/dictionaries/cancer-drug"

    for letter in alphabets:
        url = f"{base_url}/expand/{letter}"
        new_url = f"{new_base_url}/expand/{letter}"
        articles.append(url)
        articles.append(new_url)

    file_path = './cancer_types/output.json'

    # file_path_2 = './cancer_types/cancer_types_links.json'

    # Read JSON data from the file
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # with open(file_path_2, 'r') as file:
    #     json_data_2  = json.load(file)

    # # Extract links and append to the existing array
    # new_links = [item['link'] for item in json_data]
    # articles.extend(new_links)

    # # Iterate through the dictionary and extend the existing list with the links
    # for letter, links in json_data_2.items():
    #     articles.extend(links)




    # Scrapes the blogs above
    loader = AsyncChromiumLoader(articles)

    docs = loader.load()



    # Converts HTML to plain text 
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    if os.path.isfile('report.txt'):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                        chunk_overlap=200)
        chunked_documents = text_splitter.split_documents(docs_transformed)

        db = FAISS.from_documents(chunked_documents, 
                                HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                                model_kwargs={'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}, encode_kwargs={'normalize_embeddings': True}))
        
        loader =  TextLoader('report.txt')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        db.add_documents(texts)
    else:
    # Chunk text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                            chunk_overlap=200)
        chunked_documents = text_splitter.split_documents(docs_transformed)

        # Load chunked documents into the FAISS index
        db = FAISS.from_documents(chunked_documents, 
                                HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                                model_kwargs={'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}, encode_kwargs={'normalize_embeddings': True}))


    return db
    

# db = load_data()


# Path to your JSON file

def process_query_llama2(query, llm, db):
    retriever = db.as_retriever()
    llm_chain = LLMChain(llm=llm, prompt=template)
    rag_chain = ( 
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )
    ans = rag_chain.invoke(query)
    print(ans)
    return process_llm_response(ans)
