import streamlit as st
import numpy as np
import re
from utils.esg_chain_wx import GenerateStandardChain,GenerateEsgChain,framework,get_collection_list, vectorDB,TranslateChain,Generate
# from utils.esg_chain import GenerateStandardChain,GenerateEsgChain,framework,get_collection_list, vectorDB,TranslateChain,Generate


import os
from utils.pdf2doc import extract_text_table, toDocuments, embeddings
from langchain.vectorstores import Chroma

st.title("ESG 報告建議")
chapter = st.selectbox("關於GRI, 你想查詢哪個章節的範本",set(framework().keys()))
PDF_FOLDER=os.getenv("UPLOAD_FOLDER","/app/pdfs/ESG")
if not os.path.isdir(PDF_FOLDER): 
    os.makedirs(PDF_FOLDER) 
with st.form("my-form", clear_on_submit=True):
    uploaded_file = st.file_uploader("FILE UPLOADER")
    submitted = st.form_submit_button("UPLOAD!")

    if submitted and uploaded_file is not None:
        st.write("UPLOADED!")
        collection_name = uploaded_file.name.split('/')[-1].split('.')[0]
        bytes_data = uploaded_file.getvalue()
        fname_pdf = os.path.join(PDF_FOLDER,uploaded_file.name)
        with open(fname_pdf,"wb") as f:
            f.write(bytes_data)
        
            extracted = extract_text_table(fname_pdf)
            docstore = Chroma.from_documents(
                    documents=toDocuments(extracted['text'],token_limit=300),
                    embedding=embeddings,
                    collection_name=collection_name,
                    persist_directory=os.environ.get("INDEX_NAME")
                )
        os.remove(fname_pdf)
        uploaded_file = None
collection = st.sidebar.selectbox('參考文件',set(get_collection_list()))


generate = st.button("生成建議")
st.write("Click to generate!")

if generate:
    with st.container():
        vector_gri = vectorDB("GRI")
        qa_chain = GenerateStandardChain(vector_gri.vectorstore())

        vector_esg = vectorDB(collection)

    # You can call any Streamlit command, including custom components:
        res = TranslateChain(GenerateEsgChain(user_prompt=framework()[chapter],qa_chain=qa_chain,vector_instance=vector_esg.vectorstore()))
        
        st.markdown("### ESG Report Suggestion:")
        if res.strip().endswith("。"):
            
            st.markdown(res)
        else:
            st.markdown(res+Generate("complete the following text in Markdown format:\n"+res))
        
        st.markdown("### Retrieved GRI Standard:")
        with st.expander("查看參考來源"):
            source_document = "".join([f"### 文件 {str(index+1)}:\n {d.page_content}\n\n" for index,d in enumerate(vector_esg.vectorstore().similarity_search(framework()[chapter],k=3))])
            
            st.markdown(f"""
                {source_document}
                """)
with st.container():
        st.header("ESG GRI 使用者輸入章節")
        vector_gri = vectorDB("GRI")
        qa_chain = GenerateStandardChain(vector_gri.vectorstore())

        vector_esg = vectorDB(collection)
        col1, col2 = st.columns(2)
    # You can call any Streamlit command, including custom components:
        with col1:
            txt = st.text_area("ESG GRI Standard")
        if txt:
            with col2:
                st.markdown(GenerateEsgChain(user_prompt=txt,qa_chain=qa_chain,vector_instance=vector_esg.vectorstore()))
