
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from os import environ
from dotenv import load_dotenv
import json
from langchain.vectorstores import Chroma
from glob import glob
import os
from langchain.schema import format_document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain.schema import StrOutputParser
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
import chromadb

load_dotenv()
params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 30,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}


WX_MODEL = os.environ.get("WX_MODEL")
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }
llm = Model(model_id=WX_MODEL, credentials=creds, params=params, project_id=project_id).to_langchain()

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

    template = """[INST]<<SYS>>
    You'll be responsible for writing ESG (Environmental, Social, and Governance) reports, which are vital to our organization. We need to ensure that our ESG reports are accurate and transparent to meet the expectations of our shareholders and stakeholders. Let's get started.
    You'll bed provided with previous ESG reports as references. This will help you understand our style and content. 
    Our ESG reports follow specific ESG frameworks, such as GRI, SASB, etc. You are familiar with the structure and key indicators of these frameworks.
    <</SYS>>


    According to the retrieved documents, summarize Standard field with given format.

    % Documents
    {context}
    Format:
    1. Step by step instruction of how to generate report
    2. Standard index, e.g, 2.1.a or 2.1.b etc.
    3. Fields that should include in generating esg report.
    Standard: {question}
    Answer: [/INST]
    """
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


    qa_chain_esg = (
        {
            "summarize": qa_chain| vector_instance.as_retriever(search_type="mmr", search_kwargs={'k': 3})| _combine_documents,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain_esg.invoke({"question": user_prompt}) 
def TranslateChain(text):

    params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0,
    GenParams.STOP_SEQUENCES:["\n\n\n"],
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 2
}
    llm = Model(model_id="meta-llama/llama-2-70b-chat",
                credentials=creds,
                params=params,
                project_id=project_id
            ).to_langchain()
    prompt =  PromptTemplate.from_template(
    '''[INST] <<SYS>>
        You are a helpful, respectful and honest assistant.
        Always answer as helpfully as possible, while being safe.
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
        If you don't know the answer to a question, please don't share false information.
        
        Translate from English to Chinese as the following example:
        English: The Board of Directors is responsible for overseeing the bank's operations, including its financial performance, risk management, and corporate governance practices.
        Chinese: 董事會負責監督銀行的運營，包括財務績效、風險管理和公司治理實務。\n
        English: The board members' tenure is 3 years, and they can be re-elected for a maximum of 2 consecutive terms.
        Chinese: 董事會成員任期3年，最多可連任2屆。\n
        English: {original_text}
        Chinese: ''')
    trans = (
        {"original_text": itemgetter("original_text")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return trans.invoke({"original_text": text})

def Generate(prompt,stop_sequences=["。\n\n","\n\n\n"]):
    params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 30,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0,
    GenParams.STOP_SEQUENCES:stop_sequences,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
    project_id = os.getenv("PROJECT_ID", None)

    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

    WX_MODEL = os.environ.get("WX_MODEL")

    llm = Model(
                    model_id="meta-llama/llama-2-70b-chat",
                    credentials=creds,
                    params=params,
                    project_id=project_id
                ).to_langchain()
    return llm(prompt)
def framework():
    with open("/app/guildeline/GRI.json") as f:
        guideline = json.load(f)
                              
    return guideline
# generate_template("如果組織使用與法定名稱不同但眾所皆知的商業名稱時，則宜在其法定名稱外額外報導")