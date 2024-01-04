
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from os import environ
from dotenv import load_dotenv
from genai.model import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain.llm import LangChainInterface
import json
from langchain.vectorstores import Chroma
from glob import glob
import os
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceHubEmbeddings

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
import chromadb

load_dotenv()
params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stream=False,
        top_k=50,
        top_p=1,
    )


WX_MODEL = os.environ.get("WX_MODEL")
creds = Credentials(os.environ.get("BAM_API_KEY"), "https://bam-api.res.ibm.com/v1")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
            )
if HUGGINGFACEHUB_API_TOKEN!="":
    repo_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embeddings = HuggingFaceHubEmbeddings(
        task="feature-extraction",
        repo_id = repo_id,
        huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
    )
else:
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
class vectorDB():
    def __init__(self,collection) -> None:
        self.collection_name = collection
    def vectorstore(self):
        
        vectorstore = Chroma(
                        embedding_function=embeddings,
                        collection_name=self.collection_name,
                        persist_directory=os.environ.get("INDEX_NAME")
                    
            )
        return vectorstore

vectorstore_gri = Chroma(
                        embedding_function=embeddings,
                        collection_name="GRI",
                        persist_directory=os.environ.get("INDEX_NAME")
                    
            )

# qa_chain= RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3})
# )
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [f"Document {idx+1}. {format_document(doc, document_prompt)}" for idx, doc in enumerate(docs)]
    return document_separator.join(doc_strings)

def get_collection_list():
    client = chromadb.PersistentClient(path=os.environ.get("INDEX_NAME"))
    return [cols.name for cols in client.list_collections() if cols.name!="GRI"]

def GenerateStandardChain(vectorstore):

#     template = """<s>[INST] <<SYS>>
# INSTRUCTION:
# 你是一位ESG報告專家。分析ESG frameword 並用中文回答。
# Please answer in Chinese, you are the esg report advisor.You will be provided standard of ESG framework and some related documents. 
# If answer can be found exactly in Documents answer the original standard. If not exactly the same, summarize and answer it.
# Produce the answer using the steps as below.
#         Step 1: Understand the Standard.
#         Step 2: WRITE the ENGLISH answer.
#         Step 3: Translate the ENGLISH answer into Chinese language.
# AVOID the new line as much as possible.
# Start your response with 'Sure, I can answer in Chinese. Here's my response:'
# <</SYS>>
# INPUT:
# Documents: 
# {context}
# Standard: {question}
# Step 1: Understand Standard
# Step 2: ENGLISH ANSWER:
# Step 3: TRADITIONAL CHINESE TRANSLATED ANSWER:
# [/INST]
#     '''
#     """
#     prompt = PromptTemplate.from_template(template)

    qa_chain = itemgetter("question") | vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 1})| _combine_documents
    
    return qa_chain
def GenerateEsgChain(user_prompt,qa_chain,vector_instance):
    
    prompt = PromptTemplate.from_template(
        '''[INST]<<SYS>>
        Based on the past report,you should ONLY use the most relevant document 'Past Report' to summarize the key information that must contain in the report. Using following format to generate the report template.
        Format:
        Only use the 揭露項目(Standard) in STANDARD to generate answer.
        1. Title of the standard
        2. key information or essential field. 
        3. Make an example to ensure everyone understand each field.
        
        <</SYS>>

        % Past Report
        {summarize}
        % STANDARD
        {question}
        
        ESG 報告模板: [/INST]'''
    )

    # chain2 = (
    #     {"summarize": qa_chain, "question":itemgetter("question")}
    #     | prompt2
    #     | llm
    #     | StrOutputParser()
    # )
    qa_chain_esg = (
        {
            "summarize":qa_chain | vector_instance.as_retriever(search_type="mmr", search_kwargs={'k': 3})| _combine_documents,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain_esg.invoke({"question": user_prompt}) 
def TranslateChain(text):
    params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stop_sequences=["\n\n\n"],
        stream=False,
        top_k=50,
        top_p=1,
    )

    llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
            )
    prompt =  PromptTemplate.from_template("INST] <<SYS>>\n"\
    "You are a helpful, respectful and honest assistant.\n"\
    "Always answer as helpfully as possible, while being safe.\n"\
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\n"\
    "Please ensure that your responses are socially unbiased and positive in nature.\n"\
    "\n"\
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\n"\
    "If you don't know the answer to a question, please don't share false information.\n"\
    "\n"\
    "Translate from English to Chinese as the following example:\n"\
    "English: The Board of Directors is responsible for overseeing the bank's operations, including its financial performance, risk management, and corporate governance practices.\n"\
    "Chinese: 董事會負責監督銀行的運營，包括財務績效、風險管理和公司治理實務。\n\n"\
    "English: The board members' tenure is 3 years, and they can be re-elected for a maximum of 2 consecutive terms.\n"\
    "Chinese: 董事會成員任期3年，最多可連任2屆。\n\n"\
    "English: {original_text}\n"\
    "Chinese: ")
    trans = (
        {"original_text": itemgetter("original_text")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return trans.invoke({"original_text": text}) 
def Generate(prompt,stop_sequences=["。\n\n","\n\n\n"]):
    params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stop_sequences=stop_sequences,
        stream=False,
        top_k=50,
        top_p=1,
    )


    WX_MODEL = os.environ.get("WX_MODEL")
    creds = Credentials(os.environ.get("BAM_API_KEY"), "https://bam-api.res.ibm.com/v1")

    llm = LangChainInterface(
                    model=WX_MODEL,
                    credentials=creds,
                    params=params,
                )
    return llm(prompt)
def framework():
    with open("/app/guildeline/GRI.json") as f:
        guideline = json.load(f)
                              
    return guideline
# generate_template("如果組織使用與法定名稱不同但眾所皆知的商業名稱時，則宜在其法定名稱外額外報導")