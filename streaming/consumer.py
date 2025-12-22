import requests
from predict import predict_sentiment
from postgres_writer import insert_tweets

FAKE_API_URL = "http://localhost:9000/batch"

def consume_microbatch(batch_size=10):
    response = requests.get(FAKE_API_URL, params={"batch_size": batch_size})
    tweets = response.json()

    processed = []
    for tweet in tweets:
        sentiment = predict_sentiment(tweet["text"])
        processed.append({
            "airline": tweet["airline"],
            "text": tweet["text"],
            "sentiment": sentiment,
            "negativereason": tweet.get("negativereason"),
            "tweet_created": tweet["tweet_created"]
        })

    insert_tweets(processed)
    print(f"{len(processed)} tweets traités et insérés dans PostgreSQL.")

if __name__ == "__main__":
    consume_microbatch(10)
