# --- 8. 텍스트 기반 감정 분석
from transformers import pipeline
from kiwipiepy import Kiwi
from collections import Counter

# Initialize sentiment-analysis pipeline once
_sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=-1
)

# Initialize tokenizer for noun extraction
_kiwi = Kiwi()

# Stopwords for filtering nouns
_stopwords = {
    "안녕하세요", "입니다", "있습니다", "합니다", "같습니다",
    "저희", "그리고", "그러면", "그런데", "저는", "저기",
    "음", "어", "아", "~", "…"
}

def analyze_text_emotion(text: str, top_k: int = 5):
    """
    Perform sentiment analysis and noun frequency extraction on the given text.

    Args:
        text (str): Input transcript text.
        top_k (int): Number of top sentiment candidates to return.

    Returns:
        dict: {
            'label': Most probable sentiment label (e.g., '5 stars'),
            'score': Confidence score for the top label,
            'distribution': Dict of label->score for top_k candidates,
            'top_nouns': List of (noun, count) tuples for top 20 nouns
        }
    """
    # Sentiment analysis
    results = _sentiment_pipe(text, top_k=top_k)
    distribution = {r['label']: r['score'] for r in results}
    # Select top label by highest score
    top = max(results, key=lambda r: r['score'])
    label, score = top['label'], top['score']

    # Noun extraction
    tokens = _kiwi.tokenize(text)
    nouns = [word for word, pos, *_ in tokens if pos.startswith("NN")]
    filtered = [w for w in nouns if w not in _stopwords and len(w) > 1]
    noun_freq = Counter(filtered).most_common(20)

    return {
        "label": label,
        "score": score,
        "distribution": distribution,
        "top_nouns": noun_freq
    }
