import os
import requests
import warnings
import logging
import streamlit as st
from duckduckgo_search import DDGS
from huggingface_hub import InferenceClient
from transformers.utils import logging as hf_logging
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
token = os.getenv('HF_TOKEN')

def classify_fake_news(headline: str, token: str):
    api_url = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": headline}
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        result = resp.json()
        label = result[0][0]["label"]
        score = result[0][0]["score"]
    except Exception:
        label, score = "unknown", 0.0
    return label, score

def search_duckduckgo(query: str, max_results: int = 5):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
        return results
    except Exception:
        return []

def llama_reasoning(prompt: str, token: str):
    client = InferenceClient(provider="fireworks-ai", api_key=token)
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ Could not fetch Llama reasoning ({e})"

def verify_with_web(headline: str, token: str):
    query = headline
    results = search_duckduckgo(query)
    context = "\n".join([f"{r['title']}: {r['body']}" for r in results])
    prompt = f"""
Headline: {headline}

Below are search results from reliable news sources:
{context}

Based on this evidence, determine if the headline is true, false, or unverified.
Provide a short reasoning.
"""
    return llama_reasoning(prompt, token)

def analyze_sentiment(text: str, token: str):
    api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": text}
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        result = resp.json()
        label = result[0][0]["label"]
        score = result[0][0]["score"]
    except Exception:
        label, score = "NEUTRAL", 0.0
    return label, score

def detect_fake_news(headline: str, token: str):
    label, score = classify_fake_news(headline, token)
    web_summary = verify_with_web(headline, token)
    summary_lower = web_summary.lower()

    if any(word in summary_lower for word in ["false", "fake", "not true", "incorrect"]):
        verdict = "❌ Fake (verified as false)"
    elif any(word in summary_lower for word in ["true", "confirmed", "verified"]):
        verdict = "✅ Real (verified as true)"
    else:
        # fallback to sentiment
        if "negative" in sentiment_label.lower():
            verdict = "❌ Likely Fake (negative sentiment in sources)"
        else:
            verdict = "✅ Likely True (neutral or positive coverage)"

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection App")
st.caption("Powered by Llama 3.1 + DuckDuckGo + Hugging Face")
st.write("---")

headline = st.text_area("Enter a news headline to verify:", height=100)

if st.button("🔍 Verify Headline", use_container_width=True):
    if not token:
        st.error("HF_TOKEN not set in environment variables.")
    elif not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        with st.spinner("Analyzing... please wait ⏳"):
            label, score, web_summary, sentiment_label, sentiment_score, verdict = detect_fake_news(headline, token)
        st.subheader("🌐 Web Verification Summary")
        st.write(web_summary)
        st.subheader("✅ Final Verdict")
        st.success(verdict)

st.markdown("---")
st.caption("© 2025 Fake News Verifier | Built with Streamlit + Hugging Face")
