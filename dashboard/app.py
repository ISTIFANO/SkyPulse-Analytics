import streamlit as st
import psycopg2
import pandas as pd
from streamlit_autorefresh import st_autorefresh

DATABASE_URL = "postgresql://user:password@localhost:5432/aerostream"

st.set_page_config(page_title="AeroStream Dashboard", layout="wide")
st.title("✈️ AeroStream – Streaming Sentiment Analysis")

st_autorefresh(interval=60000)

conn = psycopg2.connect(DATABASE_URL)
df = pd.read_sql("SELECT * FROM tweets_stream", conn)
conn.close()

st.metric("Total Tweets", len(df))
st.metric("Airlines", df["airline"].nunique())
st.metric("Negative (%)",
          round((df[df["sentiment"] == "negative"].shape[0] / len(df)) * 100, 2)
          if len(df) > 0 else 0)

st.bar_chart(df["airline"].value_counts())
st.bar_chart(df["sentiment"].value_counts())
