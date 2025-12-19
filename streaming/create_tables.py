import psycopg2

DATABASE_URL = "postgresql://aerostream_user:aerostream_pass@postgres:5432/aerostream"

def create_tables():

    create_query = """
    CREATE TABLE IF NOT EXISTS tweets_stream (
        id SERIAL PRIMARY KEY,
        airline TEXT NOT NULL,
        text TEXT NOT NULL,
        sentiment TEXT CHECK (sentiment IN ('positive','neutral','negative')),
        negativereason TEXT,
        tweet_created TIMESTAMP,
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_airline ON tweets_stream(airline);
    CREATE INDEX IF NOT EXISTS idx_sentiment ON tweets_stream(sentiment);
    """
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(create_query)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables PostgreSQL créées avec succès !")

if __name__ == "__main__":
    create_tables()
