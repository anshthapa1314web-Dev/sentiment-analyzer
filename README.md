# AI-Based Sentiment Analyzer

A Streamlit app that analyzes reviews and comments using a Hugging Face transformer model and stores the results in MongoDB Atlas.

## Features
- Single review analysis with confidence score
- Bulk analysis via CSV upload (up to 200 rows)
- Sentiment labels: positive, neutral, negative
- Results saved to MongoDB
- Report tab with pie chart, average confidence chart and history table
- Download report as CSV

## Tech Stack
Python, Streamlit, Hugging Face Transformers (cardiffnlp/twitter-roberta-base-sentiment-latest), MongoDB Atlas (PyMongo), Pandas, Plotly

## How to Run
```
pip install -r requirements.txt
$env:MONGO_URI="your-mongodb-connection-string"
streamlit run app.py
```
The first run downloads the model (about 500 MB).

## Screenshots
![Report](screenshot.png)
![MongoDB](mongodb.png)

## Author
Ansh Thapa