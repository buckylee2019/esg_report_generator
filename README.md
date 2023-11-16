# wx-esg-report

1. Build docker image
2. `docker run -it -d  --name [NameofContainer] -v /root/wx-esg-report:/app -p 8011:8501 [NameofImage]`
3. `mkdir pdfs`
4. `mkdir pdfs/ESG pdfs/GRI`
5. Upload ESG pdfs and GRI pdfs to the directory respectly.
6. `docker exec -it [NameofContainer] python /app/utils/pdf2doc.py /app/pdfs/ESG`
7. `docker exec -it [NameofContainer] python /app/utils/pdf2doc.py /app/pdfs/GRI`
8. `docker exec -it [NameofContainer] streamlit run /app/esg_app.py`
9. Open your browser, enter http://[your ip address]:8011