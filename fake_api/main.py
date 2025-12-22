from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone
import random
from faker import Faker

app = FastAPI(
    title="SkyPulse Fake Tweet Generator API",
    description="""
## 🛫 Airline Tweet Generator API

This API generates fake airline-related tweets for testing and development purposes.

### Features:
- Generate realistic airline customer feedback tweets
- Simulate various sentiment types (positive, neutral, negative)
- Include negative reasons for complaint tweets
- Batch generation support

### Airlines Covered:
- Virgin America
- United
- Southwest
- Delta
- US Airways
- American
    """,
    version="1.0.0",
    contact={
        "name": "SkyPulse Analytics Team",
        "email": "support@skypulse.io"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_tags=[
        {
            "name": "Tweets",
            "description": "Generate fake airline tweets for testing"
        },
        {
            "name": "Health",
            "description": "API health check endpoints"
        }
    ]
)

fake = Faker()
AIRLINES = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
SENTIMENTS = ['neutral', 'positive', 'negative']
NEGATIVE_REASONS = [
    'Late Flight', 'Customer Service Issue',
    'Lost Luggage', 'Cancelled Flight'
]


class Tweet(BaseModel):
    """Schema for a generated tweet"""
    airline: str = Field(..., description="Name of the airline", example="Delta")
    negativereason: Optional[str] = Field(
        None, 
        description="Reason for negative sentiment (only present for negative tweets)",
        example="Late Flight"
    )
    tweet_created: str = Field(
        ..., 
        description="ISO 8601 timestamp when the tweet was created",
        example="2025-12-22T10:30:00+00:00"
    )
    text: str = Field(..., description="The tweet text content", example="Great flight experience!")

    class Config:
        json_schema_extra = {
            "example": {
                "airline": "Delta",
                "negativereason": None,
                "tweet_created": "2025-12-22T10:30:00+00:00",
                "text": "Amazing service on my flight today!"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="API status", example="healthy")
    timestamp: str = Field(..., description="Current server timestamp")
    version: str = Field(..., description="API version", example="1.0.0")


def generate_tweet() -> Tweet:
    """Generate a single fake tweet"""
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


@app.get("/", tags=["Health"])
def root():
    """
    Root endpoint - API welcome message
    """
    return {
        "message": "Welcome to SkyPulse Fake Tweet Generator API",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint
    
    Returns the current status and version of the API.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0"
    )


@app.get("/batch", response_model=List[Tweet], tags=["Tweets"])
def get_microbatch(
    batch_size: int = Query(
        default=10,
        ge=1,
        le=1000,
        description="Number of tweets to generate (1-1000)"
    )
):
    """
    Generate a batch of fake airline tweets
    
    - **batch_size**: Number of tweets to generate (default: 10, max: 1000)
    
    Returns a list of generated tweets with random airlines, sentiments, and text.
    """
    return [generate_tweet() for _ in range(batch_size)]


@app.get("/airlines", tags=["Tweets"])
def get_airlines():
    """
    Get list of available airlines
    
    Returns all airline names that can appear in generated tweets.
    """
    return {"airlines": AIRLINES}


@app.get("/negative-reasons", tags=["Tweets"])
def get_negative_reasons():
    """
    Get list of possible negative reasons
    
    Returns all possible reasons that can appear in negative sentiment tweets.
    """
    return {"reasons": NEGATIVE_REASONS}
