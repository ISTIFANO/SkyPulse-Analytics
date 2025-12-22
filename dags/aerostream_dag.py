from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import requests
import random
import sys
import os

sys.path.insert(0, '/opt/airflow/streaming')

FAKE_API_URL = os.getenv("FAKE_API_URL", "http://fastapi:9000/batch")
POSTGRES_CONN_ID = "aerostream_db"


def fetch_tweets(**kwargs):
    batch_size = random.choice([5, 10, 15])
    
    response = requests.get(
        FAKE_API_URL,
        params={"batch_size": batch_size},
        timeout=30
    )
    response.raise_for_status()
    tweets = response.json()
    
    print(f"Fetched {len(tweets)} tweets from API")
    kwargs['ti'].xcom_push(key='tweets', value=tweets)


def predict_sentiment(**kwargs):
    ti = kwargs['ti']
    tweets = ti.xcom_pull(task_ids='fetch_tweets', key='tweets')
    
    try:
        from predict import predict_sentiment as predict_fn
    except ImportError:
        import random
        def predict_fn(text):
            return random.choice(['positive', 'neutral', 'negative'])
    
    processed_tweets = []
    for tweet in tweets:
        sentiment = predict_fn(tweet["text"])
        processed_tweets.append({
            "airline": tweet["airline"],
            "text": tweet["text"],
            "sentiment": sentiment,
            "negativereason": tweet.get("negativereason"),
            "tweet_created": tweet["tweet_created"]
        })
    
    print(f"Predicted sentiment for {len(processed_tweets)} tweets")
    ti.xcom_push(key='processed_tweets', value=processed_tweets)


def create_table(**kwargs):
    """Create the tweets_stream table if it doesn't exist"""
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS tweets_stream (
            id SERIAL PRIMARY KEY,
            airline TEXT NOT NULL,
            text TEXT NOT NULL,
            sentiment TEXT CHECK (sentiment IN ('positive', 'neutral', 'negative')),
            negativereason TEXT,
            tweet_created TIMESTAMP,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_airline ON tweets_stream(airline);
        CREATE INDEX IF NOT EXISTS idx_sentiment ON tweets_stream(sentiment);
    """
    
    pg_hook.run(create_table_sql)
    print("Table tweets_stream created/verified successfully")


def save_to_database(**kwargs):
    """Save processed tweets to PostgreSQL"""
    ti = kwargs['ti']
    tweets = ti.xcom_pull(task_ids='predict_sentiment', key='processed_tweets')
    
    if not tweets:
        print("No tweets to save")
        return
    
    rows = [
        (
            tweet["airline"],
            tweet["text"],
            tweet["sentiment"],
            tweet.get("negativereason"),
            tweet["tweet_created"]
        )
        for tweet in tweets
    ]
    
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    pg_hook.insert_rows(
        table="tweets_stream",
        rows=rows,
        target_fields=[
            "airline",
            "text",
            "sentiment",
            "negativereason",
            "tweet_created"
        ]
    )
    
    print(f"Saved {len(rows)} tweets to PostgreSQL")



default_args = {
    'owner': 'skypulse',
    'retries': 1,
}

with DAG(
    dag_id="skypulse_streaming_pipeline",
    description="Fetch tweets, predict sentiment, and store in PostgreSQL",
    default_args=default_args,
    start_date=datetime(2025, 12, 22),
    schedule="*/1 * * * *",  # Every minute
    catchup=False,
    tags=["skypulse", "sentiment", "streaming"]
) as dag:

    t1 = PythonOperator(
        task_id="fetch_tweets",
        python_callable=fetch_tweets
    )

    t2 = PythonOperator(
        task_id="predict_sentiment",
        python_callable=predict_sentiment
    )

    t3 = PythonOperator(
        task_id="create_table",
        python_callable=create_table
    )

    t4 = PythonOperator(
        task_id="save_to_database",
        python_callable=save_to_database
    )

    t1 >> t2 >> t3 >> t4
