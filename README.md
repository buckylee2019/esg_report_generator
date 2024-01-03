# wx-esg-report

1. Build docker image 
2. `docker build . -t <Your Preferred Image Name>`
3. `docker run -it -d  --name <NameofContainer> -v <download repo path>:/app -p 8001:8001 <Your Preferred Image Name>`
4. Open your browser, enter http://[your ip address]:8001
5. Upload your PDF from browser.