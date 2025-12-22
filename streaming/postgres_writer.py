import psycopg2
from psycopg2.extras import execute_values
import os

# Use localhost:5433 for local execution, postgres:5432 for Docker
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://aerostream_user:aerostream_pass@localhost:5433/aerostream"
)

def insert_tweets(tweets):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    records = [
        (
            t["airline"], t["text"], t["sentiment"], 
            t.get("negativereason"), t["tweet_created"]
        ) for t in tweets
    ]

    execute_values(
        cur,
        """
        INSERT INTO tweets_stream
        (airline, text, sentiment, negativereason, tweet_created)
        VALUES %s
        """,
        records
    )

    conn.commit()
    cur.close()
    conn.close()
