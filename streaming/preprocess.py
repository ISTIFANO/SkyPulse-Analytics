import re

def clean_text(text: str) -> str:
 
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text) 
    text = re.sub(r"@\w+", "", text)            
    text = re.sub(r"[^a-z\s]", "", text)     
    text = text.strip()
    return text
