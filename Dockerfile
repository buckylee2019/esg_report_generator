# Use an official Python base image from the Docker Hub
FROM python:3.11-slim

# Install browsers

# Declare working directory
WORKDIR /app

COPY . .

# Install any necessary packages specified in requirements.txt.
RUN pip install -r requirements.txt

