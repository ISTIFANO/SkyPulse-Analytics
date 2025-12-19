from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import random
from faker import Faker

app = FastAPI(title="Fake Tweet Generator API")

fake = Faker()
AIRLINES = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
SENTIMENTS = ['neutral', 'positive', 'negative']
NEGATIVE_REASONS = [
    'Late Flight', 'Customer Service Issue',
    'Lost Luggage', 'Cancelled Flight'
]

class Tweet(BaseModel):
    airline: str
    negativereason: Optional[str]
    tweet_created: str
    text: str

def generate_tweet() -> Tweet:
    airline = random.choice(AIRLINES)
    sentiment = random.choice(SENTIMENTS)

    negativereason = random.choice(NEGATIVE_REASONS) if sentiment == "negative" else None
    text = fake.sentence()

    return Tweet(
        airline=airline,
        negativereason=negativereason,
        tweet_created=datetime.now(timezone.utc).isoformat(),
        text=text
    )

@app.get("/batch", response_model=List[Tweet])
def get_microbatch(batch_size: int = 10):
    return [generate_tweet() for _ in range(batch_size)]
