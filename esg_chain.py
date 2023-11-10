
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
from langchain.embeddings import HuggingFaceEmbeddings
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

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
WX_MODEL = os.environ.get("WX_MODEL")
creds = Credentials(os.environ.get("BAM_API_KEY"), "https://bam-api.res.ibm.com/v1")

llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
            )

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
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

def generate_standard_chain(vectorstore):

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
    Extracted detailed standard:[/INST]
    """
    prompt = PromptTemplate.from_template(template)

    qa_chain = (
        {
            "context": itemgetter("question") | vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 2})| _combine_documents,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain
def generate_esg_chain(user_prompt,qa_chain,vector_instance):
    
    prompt = PromptTemplate.from_template(
        '''[INST]<<SYS>>
        Based on the past report,you should ONLY use the most relevant document 'Past Report' to summarize the key information that must contain in the report. Using following format to generate the report template.
        Format:
        Only use the 揭露項目 in STANDARD to generate answer.
        1. Title of the standard
        2. key information or essential field. 
        3. Add example to essure everyone understand the field.
        
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
            "summarize": qa_chain| vector_instance.as_retriever(search_type="mmr", search_kwargs={'k': 2})| _combine_documents,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain_esg.invoke({"question": user_prompt}) 
def translate_chain(text):
    prompt =  PromptTemplate.from_template("""
    English: The Board of Directors is responsible for overseeing the bank's operations, including its financial performance, risk management, and corporate governance practices.
    Chinese: 董事會負責監督銀行的運營，包括財務績效、風險管理和公司治理實務。

    English: The board members' tenure is 3 years, and they can be re-elected for a maximum of 2 consecutive terms.
    Chinese: 董事會成員任期3年，最多可連任2屆。

    English: {original_text}
    Chinese: 
    """)
    trans = (
        {"original_text": itemgetter("original_text")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return trans.invoke({"original_text": text}) 
def framework():
    guideline = {
        "2.6 活動、價值鏈和其他商業關係":"""揭露項目 2-6 活動、價值鏈和其他商業關係
                組織應:
                a. 報導其現行的行業;
                b. 描述其價值鏈，包括:
                i. 組織活動、產品、服務 ，以及提供服務的市場;
                ii. 組織的供應鏈;
                iii. 組織下游的實體及其活動;
                c. 報導其他相關的商業關係;
                d. 描述與前一報導期間相比，2-6-a、2-6-b、2-6-c的顯著變化。
                """,
        "2.7 員工":"""揭露項目 2-7 員工
                組織應:
                a. 報導員工總數，以及按性別及地區分類的總數;
                b. 報導以下總數:
                i. 依性別及地區分類的永久聘雇員工;
                ii. 依性別及地區分類的臨時員工;
                iii. 依性別及地區分類的無時數保證的員工;
                iv. 依性別及地區分類的全職員工;
                v. 依性別及地區分類的兼職員工;
                c. 描述彙編數據的方法和假設，包括是否報導了以下數據:
                i. 以人數、全時等量法(full-time equivalent，FTE)或使用其他方法;
                ii. 報導期間結束日當天的數值、整個報導期間的平均值，或使用其他方法;
                d. 報導為理解2-7-a和2-7-b中報導的數據所需的脈絡資訊;
                e. 描述此報導期間內和不同報導期間之間，員工人數的顯著波動。
                """,
        "2.8 非員工的工作者":"""揭露項目 2-8 非員工的工作者
                組織應:
                a. 報導非員工的工作者，且其工作由組織控制的工作者總數，並描述:
                i. 最常見的工作者類型及其與組織的契約關係;
                ii. 其執行的工作類型;
                b. 描述彙編數據的方法和假設，包括是否報導了非員工的工作者數據:
                i. 人數、全時等量法(FTE)或使用其他方法;
                ii. 報導期間結束日當天的數值、整個報導期間的平均值，或使用其他方法;
                c. 描述此報導期間內和不同報導期間之間，非員工的工作者人數的顯著波動。""",
        "2.9 治理結構及組成":"""揭露項目 2-9 治理結構及組成
            組織應:
            a. 描述其治理結構，包括最高治理單位的委員會;
            b. 列出負責決策和監督管理組織對經濟、環境和人群衝擊的最高治理單位的委員會;
            c. 描述最高治理單位及其委員會的組成:
            i. 執行董事及非執行董事;
            ii. 獨立董事;
            iii. 治理單位成員的任期;
            iv. 治理單位各成員的其他重要職位及承諾之數目、承諾的性質;
            v. 性別;
            vi. 弱勢社會群體;
            vii. 與組織衝擊相關之能力;
            viii. 利害關係人代表。"""

    }
    return guideline
# generate_template("如果組織使用與法定名稱不同但眾所皆知的商業名稱時，則宜在其法定名稱外額外報導")