# Use an official Python base image from the Docker Hub
FROM python:3.11-slim

# Install browsers
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install -y curl wget git
# Declare working directory

WORKDIR /app

COPY ./utils ./utils
COPY esg_app.py esg_app.py
COPY requirements.txt requirements.txt



# Install any necessary packages specified in requirements.txt.
RUN pip install -r requirements.txt

# Install sentence-transformers if you don't have hugging face api token
RUN pip install sentence-transformers==2.2.2
EXPOSE 8001

ENTRYPOINT ["streamlit", "run", "/app/esg_app.py", "--server.port=8001", "--server.address=0.0.0.0"]