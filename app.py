# app.py
import os
import requests
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv

# LangChain DuckDuckGo wrapper (utility)
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# Groq API client (instead of HF)
from groq import Groq

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------- Functions -----------------

def classify_fake_news(headline: str):
    """
    Calls a Hugging Face inference endpoint for fake-news classification.
    (Kept via requests, but token not required if public)
    """
    api_url = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"
    payload = {"inputs": headline}
    try:
        resp = requests.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result and isinstance(result[0], list):
            best = max(result[0], key=lambda x: x.get("score", 0.0))
            label = best.get("label", "unknown")
            score = float(best.get("score", 0.0))
            return label, score
        return "unknown", 0.0
    except Exception:
        return "unknown", 0.0


def analyze_sentiment(text: str):
    """
    Lightweight sentiment check using a public HF model.
    """
    api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    payload = {"inputs": text}
    try:
        resp = requests.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result and isinstance(result[0], list):
            best = max(result[0], key=lambda x: x.get("score", 0.0))
            label = best.get("label", "NEUTRAL")
            score = float(best.get("score", 0.0))
            return label, score
        return "NEUTRAL", 0.0
    except Exception:
        return "NEUTRAL", 0.0


def llama_reasoning(prompt: str):
    """
    Uses Groq's Llama 3.1 model for reasoning.
    """
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set."

    client = Groq(api_key=GROQ_API_KEY)
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Llama reasoning unavailable ({e})"


# ------ LangChain DuckDuckGo search wrapper ------

ddg = DuckDuckGoSearchAPIWrapper()  # no keys needed

def search_duckduckgo_langchain(query: str, max_results: int = 5):
    try:
        raw = ddg.run(query)
        results = []
        if hasattr(ddg, "results") and ddg.results:
            for r in ddg.results[:max_results]:
                results.append({"title": r.get("title", ""), "body": r.get("snippet", "") or r.get("body", "")})
        else:
            results.append({"title": query, "body": raw})
        return results[:max_results]
    except Exception:
        return []


def verify_with_web(headline: str, max_results: int = 5):
    results = search_duckduckgo_langchain(headline, max_results=max_results)
    hit_count = len(results)

    if hit_count == 0:
        prompt = f"""
    You are a fact verification assistant.

    Headline:
    {headline}

    No relevant news sources were found.

    Question:
    Even though no direct sources were found, use your general world knowledge to give a short, reasoned guess (1–2 sentences) on whether the headline sounds plausible, fake, or unverifiable.
    """
        reasoning = llama_reasoning(prompt)
        web_summary = f"No direct sources found online.\n\nLLM reasoning:\n{reasoning}"
        return web_summary, results, hit_count


    lines = []
    for r in results:
        title = r.get("title", "") or ""
        body = r.get("body", "") or ""
        lines.append(f"Title: {title}\nSnippet: {body}\n---")
    context = "\n".join(lines)

    prompt = f"""
You are a fact verification assistant.

Headline:
{headline}

Search results (top {hit_count}):
{context}

Question:
Based on the search results above, determine whether the headline is TRUE, FALSE, or UNVERIFIED. 
Give a short reasoning and end with one of the labels: TRUE / FALSE / UNVERIFIED.
"""
    reasoning = llama_reasoning(prompt)
    top_titles = "\n".join([f"- {r.get('title','').strip()}" for r in results if r.get('title')])
    web_summary = f"Search hit count: {hit_count}\nTop titles:\n{top_titles}\n\nLLM reasoning:\n{reasoning}"
    return web_summary, results, hit_count


def compute_final_verdict(label, score, hit_count, llama_summary):
    fusion_score = 0.5

    if label == "LABEL_1":  # real
        fusion_score += score * 0.5
    else:  # fake
        fusion_score -= score * 0.5

    if hit_count > 0:
        fusion_score += 0.25

    summary_lower = llama_summary.lower()
    if "true" in summary_lower or "confirmed" in summary_lower:
        fusion_score += 0.2
    elif "false" in summary_lower or "fake" in summary_lower:
        fusion_score -= 0.2

    fusion_score = max(0.0, min(1.0, fusion_score))

    if fusion_score >= 0.75:
        verdict = f"✅ Real (fusion score={fusion_score:.2f})"
    elif fusion_score <= 0.45:
        verdict = f"❌ Fake (fusion score={fusion_score:.2f})"
    else:
        verdict = f"🤔 Ambiguous (fusion score={fusion_score:.2f})"

    return verdict


def detect_fake_news(headline: str):
    label, score = classify_fake_news(headline)
    web_summary, results, hit_count = verify_with_web(headline)
    sentiment_label, sentiment_score = analyze_sentiment(web_summary)
    summary_lower = web_summary.lower()
    words = set(summary_lower.replace("\n", " ").replace(".", " ").split())
    verdict = compute_final_verdict(sentiment_label, sentiment_score, hit_count, summary_lower)
    if "unverified" in words or "no" in summary_lower and "relevant" in summary_lower:
        verdict = "⚠️ Unverified (no strong evidence online)"
    return label, score, web_summary, sentiment_label, sentiment_score, verdict


# ------ Streamlit UI ------

st.set_page_config(page_title="Fake News Detection (Groq + LangChain + DDG)", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection App — LangChain + DuckDuckGo + Groq Llama")
st.caption("Verification powered by LangChain, DuckDuckGo Search, and Groq Llama-3")

st.write("---")
st.write("Enter a headline and click **Verify Headline**.")

headline = st.text_area("Enter a news headline to verify:", height=120)

col1, col2 = st.columns([1, 1])

with col1:
    max_results = st.number_input("Max search results", min_value=1, max_value=10, value=5, step=1)

with col2:
    show_evidence = st.checkbox("Show top search evidence titles", value=True)

if st.button("🔍 Verify Headline", use_container_width=True):
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set in environment variables. Add it to .env as GROQ_API_KEY=your_api_key")
    elif not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        with st.spinner("Analyzing... this may take a few seconds ⏳"):
            label, score, web_summary, sentiment_label, sentiment_score, verdict = detect_fake_news(headline)

        st.subheader("🧠 Model (classifier) output")
        st.write(f"Label: **{label}** — Score: **{score:.3f}**")

        st.subheader("🌐 Web Verification Summary")
        st.text(web_summary)

        if show_evidence:
            results_struct = search_duckduckgo_langchain(headline, max_results=max_results)
            if results_struct:
                st.markdown("**Top search results (titles + snippets):**")
                for i, r in enumerate(results_struct, start=1):
                    title = r.get("title", "<no title>").strip()
                    body = r.get("body", "").strip()
                    st.write(f"**{i}. {title}**")
                    if body:
                        st.write(f"> {body}")
            else:
                st.info("No structured search results available to display.")

        st.subheader("📊 Sentiment on web reasoning")
        st.write(f"{sentiment_label} (score: {sentiment_score:.3f})")

        st.subheader("✅ Final Verdict")
        if verdict.startswith("✅"):
            st.success(verdict)
        elif verdict.startswith("❌"):
            st.error(verdict)
        elif verdict.startswith("⚠️"):
            st.warning(verdict)
        else:
            st.info(verdict)

st.markdown("---")
st.caption("© 2025 Fake News Verifier | LangChain + DuckDuckGo + Groq Llama-3")
