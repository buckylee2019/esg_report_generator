# watsonx ESG Report Template Generator

## Prerequisite: 

1. IBM api key
2. watsonx.ai project ID
3. (optional) Hugging Face Token 
   If you have Hugging Face Token, delete the pip install sentence-transformer (for quick build)



## Step:
1. Build docker image 
2. `docker build . -t <Your Preferred Image Name>`
3. Add your focused standard to guideline/GRI.json
4. `docker run -it -d  --name <NameofContainer> -v <download repo path>:/app -p 8001:8001 <Your Preferred Image Name>`
5. Open your browser, enter http://[your ip address]:8001
6. Upload your PDF from browser.
   