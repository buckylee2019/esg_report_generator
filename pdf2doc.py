# STEP 1
# import libraries
import fitz
import os
import json
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Milvus
from dotenv import load_dotenv
from glob import glob
import io
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = "。\n\n",
    chunk_size = 200,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
MINIO_OBJECT_URL = os.environ.get("MINIO_OBJECT_URL")
MINIO_PUBLIC_URL = os.environ.get("MINIO_PUBLIC_IP")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
INDEX_NAME = os.getenv("INDEX_NAME")

PARENT_IMG_DIR = "/app/data/images"

repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# hf = HuggingFaceHubEmbeddings(
#     task="feature-extraction",
#     repo_id = repo_id,
#     huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
# )



# STEP 2
# file path you want to extract images from


def extract_text_table(file):


    # open the file
    pdf_file = fitz.open(file)

    tables = []
    texts = []
    # STEP 3
    # iterate over PDF pages
    for page_index in range(len(pdf_file)):

        # get the page itself
        page = pdf_file[page_index]
        try:
            tabs = page.find_tables()
        except:
            tabs = []
        # printing number of images found in this page
        print("Page No.: ",page_index)
            # metadata = ({'image_source': "", 'page':page_index+1})
            # documents.append(Document(page_content=page.get_text(),metadata=metadata))
        for i, table in enumerate(tabs):

            # get the XREF of the image
            
            tables.append(json.dumps(table.extract(),ensure_ascii=False))
        texts.append(page.get_text())


    return {"text":texts, "table":tables}

def toDocuments(documents):
    langchain_doc = []
    for doc in documents:
        if len(doc) > 300:
            docs = text_splitter.create_documents([doc])
            langchain_doc.extend(docs)
        else:
            langchain_doc.append(Document(page_content=doc))
    return langchain_doc
# hf2 = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")

# index = Chroma.from_documents(
#         documents=documents,
#         embedding=hf,
#         collection_name=INDEX_NAME,
#         persist_directory=INDEX_NAME
#     )
PDF_DIR = "/Users/buckylee/Documents/github/watsonxai-foundations-class/self-guided-labs/level-2/React/pdfs/GRI/"
INDEXED = False
for pdf in glob(PDF_DIR+"*.pdf"):
    
    collection_name = "GRI"
    
    if not INDEXED:
        extracted = extract_text_table(pdf)
        docstore = Chroma.from_documents(
                documents=toDocuments(extracted['text']),
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=os.environ.get("INDEX_NAME")
            )
    else:
        docstore = Chroma(
                embedding_function=embeddings,
                collection_name=collection_name,
                persist_directory=os.environ.get("INDEX_NAME")
            )


print(docstore.similarity_search("報導總部的所在位置:總部指的是一個組織的行政中心，其控制和指引組織本身。"))


