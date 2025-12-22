import joblib
from sentence_transformers import SentenceTransformer
from preprocess import clean_text

MODEL_PATH = "models/voting_classifier.joblib"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

embedder = SentenceTransformer(EMBEDDING_MODEL)
classifier = joblib.load(MODEL_PATH)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    prediction = classifier.predict(embedding)[0]
    # Convert numpy int to string sentiment
    if isinstance(prediction, (int, float)) or hasattr(prediction, 'item'):
        prediction = int(prediction) if hasattr(prediction, 'item') else prediction
        return LABEL_MAP.get(prediction, str(prediction))
    return str(prediction)
