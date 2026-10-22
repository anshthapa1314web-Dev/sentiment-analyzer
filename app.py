import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient
from transformers import pipeline

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("💬 AI-Based Sentiment Analyzer")
st.caption("Hugging Face (RoBERTa) + MongoDB + Streamlit")


@st.cache_resource(show_spinner="Model load ho raha hai (pehli baar download hoga)...")
def load_model():
    return pipeline("sentiment-analysis", model=MODEL)


@st.cache_resource
def get_collection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client["sentiment_db"]["reviews"]
    except Exception:
        return None


model = load_model()
col = get_collection()

if "history" not in st.session_state:
    st.session_state.history = []

if col is None:
    st.warning("MongoDB connect nahi hua. Results sirf is session mein rahenge.")
else:
    st.success("MongoDB connected ✅")


def analyze(texts):
    outputs = model(texts, truncation=True, max_length=512)
    now = datetime.now()
    return [
        {
            "text": t,
            "sentiment": o["label"].lower(),
            "score": round(float(o["score"]), 4),
            "created_at": now,
        }
        for t, o in zip(texts, outputs)
    ]


def save(rows):
    st.session_state.history.extend(rows)
    if col is not None and rows:
        col.insert_many([dict(r) for r in rows])


def load_history():
    if col is not None:
        data = list(col.find({}, {"_id": 0}).sort("created_at", -1).limit(1000))
    else:
        data = list(reversed(st.session_state.history))
    return pd.DataFrame(data)


tab1, tab2, tab3 = st.tabs(["Single Review", "Bulk CSV", "Report"])

with tab1:
    text = st.text_area("Review / comment likho", height=120)
    if st.button("Analyze", key="single"):
        if text.strip():
            row = analyze([text.strip()])
            save(row)
            r = row[0]
            c1, c2 = st.columns(2)
            c1.metric("Sentiment", r["sentiment"].upper())
            c2.metric("Confidence", f"{r['score'] * 100:.1f}%")
        else:
            st.info("Pehle kuch likho.")

with tab2:
    st.write("CSV upload karo jisme reviews ka ek column ho (max 200 rows).")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        column = st.selectbox("Review wala column", df.columns)
        if st.button("Analyze all", key="bulk"):
            texts = df[column].dropna().astype(str).head(200).tolist()
            with st.spinner("Analyze ho raha hai..."):
                rows = analyze(texts)
                save(rows)
            st.success(f"{len(rows)} reviews analyze aur save ho gaye. Report tab dekho.")
            st.dataframe(pd.DataFrame(rows).drop(columns=["created_at"]))

with tab3:
    hist = load_history()
    if hist.empty:
        st.info("Abhi koi data nahi hai. Pehle kuch reviews analyze karo.")
    else:
        colors = {"positive": "#2ca02c", "neutral": "#7f7f7f", "negative": "#d62728"}
        counts = hist["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]

        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.pie(counts, names="sentiment", values="count", color="sentiment",
                   color_discrete_map=colors, title="Sentiment Distribution"),
            width="stretch",
        )
        avg = hist.groupby("sentiment")["score"].mean().reset_index()
        c2.plotly_chart(
            px.bar(avg, x="sentiment", y="score", color="sentiment",
                   color_discrete_map=colors, title="Average Confidence"),
            width="stretch",
        )
        st.subheader("History")
        st.dataframe(hist, width="stretch")
        st.download_button("Download CSV", hist.to_csv(index=False), "sentiment_report.csv")