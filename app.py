# app.py
import os
import requests
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv

# LangChain DuckDuckGo wrapper (utility)
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# HF Inference client (used for Llama reasoning via your chosen provider)
from huggingface_hub import InferenceClient
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") 

def classify_fake_news(headline: str, token: str):
    """
    Calls the Hugging Face inference API for the fake-news classifier.
    Picks the label with the highest score to avoid 'first element' trap.
    """
    api_url = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"inputs": headline}
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result and isinstance(result[0], list):
            best = max(result[0], key=lambda x: x.get("score", 0.0))
            label = best.get("label", "unknown")
            score = float(best.get("score", 0.0))
            return label, score
        return "unknown", 0.0
    except Exception as e:
        return "unknown", 0.0

def analyze_sentiment(text: str, token: str):
    """
    Lightweight sentiment check on the web summary (optional).
    """
    api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"inputs": text}
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
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

def llama_reasoning(prompt: str, token: str):
    client = InferenceClient(provider="fireworks-ai", api_key=token)
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300
        )
        # Return the assistant's message content
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ Llama reasoning unavailable ({e})"

# ------ LangChain DuckDuckGo search wrapper ------

ddg = DuckDuckGoSearchAPIWrapper()  # no keys needed

def search_duckduckgo_langchain(query: str, max_results: int = 5):
    """
    Uses LangChain's DuckDuckGoSearchAPIWrapper to fetch search results.
    Returns a list of result dicts, each with 'title' and 'snippet' (LangChain naming).
    """
    try:
        raw = ddg.run(query)  # run returns a string summary; for structured data we can call run_and_parse? But wrapper often returns text.
        # The wrapper's run() often returns a combined text. We still want structured results.
        # Use the underlying .get_results if present (fallback to parsing).
        results = []
        try:
            # Some LangChain versions provide .results
            if hasattr(ddg, "results") and ddg.results:
                for r in ddg.results[:max_results]:
                    results.append({"title": r.get("title", ""), "body": r.get("snippet", "") or r.get("body", "")})
            else:
                # Fallback: treat raw text as single summary
                results.append({"title": query, "body": raw})
        except Exception:
            results.append({"title": query, "body": raw})
        return results[:max_results]
    except Exception:
        # safe fallback to empty
        return []

# ------ Verification and decision logic ------

def verify_with_web(headline: str, token: str, max_results: int = 5):
    """
    Searches web (via LangChain/DDG wrapper), builds a summary and a structured 'evidence' list.
    Returns: (web_summary_text, results_list, hit_count)
    """
    results = search_duckduckgo_langchain(headline, max_results=max_results)
    hit_count = len(results)

    if hit_count == 0:
        web_summary = "No relevant news sources found."
        return web_summary, results, hit_count

    # Build context text for LLM, and a short structured list for UI
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
Based on the search results above, determine whether the headline is TRUE (supported by reliable sources), FALSE (contradicted by reliable sources), or UNVERIFIED (no reliable evidence either supporting or contradicting). 
Provide a short reasoning (1-2 sentences) and finish with one of the labels: TRUE / FALSE / UNVERIFIED.
"""
    reasoning = llama_reasoning(prompt, token)
    # Create compact web_summary we can show in UI (include reasoning + top titles)
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

    # clamp
    fusion_score = max(0.0, min(1.0, fusion_score))

    if fusion_score >= 0.75:
        verdict = f"✅ Real (fusion score={fusion_score:.2f})"
    elif fusion_score <= 0.45:
        verdict = f"❌ Fake (fusion score={fusion_score:.2f})"
    else:
        verdict = f"🤔 Ambiguous (fusion score={fusion_score:.2f})"

    return verdict




def detect_fake_news(headline: str, token: str):
    """
    Full pipeline: classifier + web verification + sentiment fallback + final verdict.
    Returns: label, score, web_summary_text, sentiment_label, sentiment_score, verdict
    """
    # 1) HF classifier (best-label)
    label, score = classify_fake_news(headline, token)

    # 2) Web verification via LangChain DDG
    web_summary, results, hit_count = verify_with_web(headline, token)

    # 3) Sentiment on the reasoning text (optional)
    sentiment_label, sentiment_score = analyze_sentiment(web_summary, token)

    # 4) Decision logic:
    summary_lower = web_summary.lower()
    # tokenize to words for safer exact-word checks
    words = set(summary_lower.replace("\n", " ").replace(".", " ").split())

    verdict = compute_final_verdict(sentiment_label,sentiment_score, hit_count, summary_lower)
    # exact check for unverified first
    if "unverified" in words or "no" in summary_lower and "relevant" in summary_lower:
        verdict = "⚠️ Unverified (no strong evidence online)"
    return label, score, web_summary, sentiment_label, sentiment_score, verdict

# ------ Streamlit UI ------

st.set_page_config(page_title="Fake News Detection (LangChain + DDG)", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection App — LangChain + DuckDuckGo")
st.caption("Verification powered by LangChain's DuckDuckGo wrapper + HF models")

st.write("---")
st.write("Enter a headline and click **Verify Headline**.")

headline = st.text_area("Enter a news headline to verify:", height=120)

col1, col2 = st.columns([1, 1])

with col1:
    max_results = st.number_input("Max search results", min_value=1, max_value=10, value=5, step=1)

with col2:
    show_evidence = st.checkbox("Show top search evidence titles", value=True)

if st.button("🔍 Verify Headline", use_container_width=True):
    if not HF_TOKEN:
        st.error("HF_TOKEN not set in environment variables. Add it to .env as HF_TOKEN=hf_xxx")
    elif not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        with st.spinner("Analyzing... this may take a few seconds ⏳"):
            label, score, web_summary, sentiment_label, sentiment_score, verdict = detect_fake_news(headline, HF_TOKEN)

        # Show classifier output
        st.subheader("🧠 Model (classifier) output")
        st.write(f"Label: **{label}** — Score: **{score:.3f}**")

        # Web verification summary and evidence
        st.subheader("🌐 Web Verification Summary")
        st.text(web_summary)

        if show_evidence:
            # If results had structured info, show them nicely
            # detect titles from web_summary/results
            # We'll attempt to parse the 'results' from verify_with_web (we returned them earlier in detection pipeline)
            _, _, results, = None, None, None  # placeholder (we don't have direct results here)
            # Instead, re-run search to provide structured evidence (cheap)
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

        # Sentiment
        st.subheader("📊 Sentiment on web reasoning")
        st.write(f"{sentiment_label} (score: {sentiment_score:.3f})")

        # Final verdict
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
st.caption("© 2025 Fake News Verifier | LangChain + DuckDuckGo + Hugging Face")
