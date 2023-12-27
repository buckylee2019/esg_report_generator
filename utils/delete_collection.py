from dotenv import load_dotenv
import chromadb
import os

load_dotenv()
client = chromadb.PersistentClient(path="/Users/buckylee/Documents/github/watsonxai-foundations-class/self-guided-labs/level-2/esg_report/ESG_REPORT")
for  cols in client.list_collections():
    
    client.delete_collection(name=cols.name)
