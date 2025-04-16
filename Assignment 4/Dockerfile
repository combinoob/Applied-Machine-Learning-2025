# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY score.py app.py ./  
COPY model ./model

# Install dependencies
RUN pip install flask nltk scikit-learn

# Download necessary nltk data
RUN python -m nltk.downloader punkt stopwords wordnet

# Expose the port the app runs on
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
