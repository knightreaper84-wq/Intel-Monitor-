"""
Global Intelligence Monitor – Real‑time OSINT Dashboard
Deployable on Streamlit Cloud / Hugging Face Spaces
"""

import os
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from geopy.geocoders import Nominatim, OpenCage
from textblob import TextBlob

# Optional NLP library
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    st.warning("spaCy not loaded – entity extraction disabled. Run 'python -m spacy download en_core_web_sm'")

# ---- CONFIGURATION ----
REFRESH_INTERVAL = 30  # seconds
MAX_EVENTS = 500

# ---- SECRETS / API KEYS ----
# Use st.secrets in production, fallback to env vars for local testing
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except:
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

try:
    OPENCAGE_KEY = st.secrets["OPENCAGE_KEY"]
except:
    OPENCAGE_KEY = os.getenv("OPENCAGE_KEY", "")

# Geocoder
if OPENCAGE_KEY:
    geolocator = OpenCage(api_key=OPENCAGE_KEY)
else:
    geolocator = Nominatim(user_agent="global_intel_monitor")

# ---- DATA FETCHERS (all public / free APIs) ----
def fetch_newsapi() -> List[Dict]:
    if not NEWS_API_KEY:
        return []
    keywords = ["earthquake", "explosion", "protest", "attack", "flood", "cyber", "crash", "fire", "riot"]
    params = {
        "q": " OR ".join(keywords),
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
        "pageSize": 15
    }
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        articles = resp.json().get("articles", [])
        events = []
        for art in articles:
            text = f"{art['title']} {art['description']}"
            events.append({
                "source": "NewsAPI",
                "title": art['title'],
                "description": art['description'][:200] if art['description'] else "",
                "published": art['publishedAt'],
                "location": extract_location(text),
                "url": art['url'],
                "raw_text": text,
                "type": "news",
                "severity": 5,
                "lat": None,
                "lon": None
            })
        return events
    except:
        return []

def fetch_usgs() -> List[Dict]:
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_hour.geojson"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        events = []
        for feat in data["features"]:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]
            events.append({
                "source": "USGS",
                "title": f"Earthquake M{props['mag']}",
                "description": f"Magnitude {props['mag']} at {props['place']}",
                "published": datetime.fromtimestamp(props["time"]/1000).isoformat(),
                "location": props['place'],
                "url": props['url'],
                "raw_text": f"earthquake magnitude {props['mag']} {props['place']}",
                "type": "disaster",
                "severity": min(props['mag'], 10),
                "lat": coords[1],
                "lon": coords[0]
            })
        return events
    except:
        return []

def fetch_opensky() -> List[Dict]:
    url = "https://opensky-network.org/api/states/all"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        states = data.get("states", [])[:15]
        events = []
        for s in states:
            if s[5] is None or s[6] is None:
                continue
            events.append({
                "source": "OpenSky",
                "title": f"Aircraft {s[1]} ({s[0]})",
                "description": f"Altitude: {s[7]}m, Speed: {s[9]}m/s",
                "published": datetime.now().isoformat(),
                "location": f"{s[6]:.2f}, {s[5]:.2f}",
                "url": "https://opensky-network.org",
                "raw_text": "aviation traffic",
                "type": "aviation",
                "severity": 3,
                "lat": s[6],
                "lon": s[5]
            })
        return events
    except:
        return []

def fetch_economic() -> List[Dict]:
    events = []
    try:
        import yfinance as yf
        tickers = ["^GSPC", "GC=F", "BTC-USD"]
        for ticker in tickers:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                latest = data["Close"].iloc[-1]
                prev = data["Close"].iloc[0] if len(data) > 1 else latest
                change = ((latest - prev) / prev) * 100
                events.append({
                    "source": "Yahoo Finance",
                    "title": f"{ticker} Update",
                    "description": f"{latest:.2f} ({change:+.2f}%)",
                    "published": datetime.now().isoformat(),
                    "location": "Global",
                    "url": f"https://finance.yahoo.com/quote/{ticker}",
                    "raw_text": f"market {ticker} change {change}%",
                    "type": "economic",
                    "severity": min(abs(change)/2, 10),
                    "lat": None,
                    "lon": None
                })
    except:
        pass
    return events

def fetch_rss() -> List[Dict]:
    import feedparser
    feeds = [
        "http://rss.cnn.com/rss/edition.rss",
        "http://feeds.bbci.co.uk/news/world/rss.xml"
    ]
    events = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:
            text = f"{entry.title} {entry.summary}"
            events.append({
                "source": feed_url.split("/")[2],
                "title": entry.title,
                "description": entry.summary[:200],
                "published": entry.get("published", datetime.now().isoformat()),
                "location": extract_location(text),
                "url": entry.link,
                "raw_text": text,
                "type": "news",
                "severity": 5,
                "lat": None,
                "lon": None
            })
    return events

def extract_location(text: str) -> str:
    if nlp is None:
        return "Unknown"
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text
    return "Unknown"

def geocode_event(event: Dict) -> Dict:
    if event.get("lat") and event.get("lon"):
        return event
    loc_name = event.get("location", "")
    if loc_name and loc_name != "Unknown":
        try:
            loc = geolocator.geocode(loc_name, timeout=3)
            if loc:
                event["lat"] = loc.latitude
                event["lon"] = loc.longitude
        except:
            pass
    return event

def compute_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

# ---- Main collection ----
def fetch_all_sources() -> pd.DataFrame:
    all_events = []
    all_events.extend(fetch_newsapi())
    all_events.extend(fetch_usgs())
    all_events.extend(fetch_opensky())
    all_events.extend(fetch_economic())
    all_events.extend(fetch_rss())

    # Deduplicate by title
    seen = set()
    unique = []
    for e in all_events:
        if e["title"] not in seen:
            seen.add(e["title"])
            unique.append(e)

    # Enrich
    for e in unique:
        e["sentiment"] = compute_sentiment(e["raw_text"])
        e = geocode_event(e)
        # Add a human‑readable timestamp
        e["timestamp"] = datetime.now().strftime("%H:%M:%S")

    df = pd.DataFrame(unique)
    if not df.empty:
        df["timestamp_dt"] = pd.to_datetime(df["published"], errors="coerce")
        df = df.sort_values("timestamp_dt", ascending=False)
    return df

# ---- STREAMLIT UI with auto-refresh ----
st.set_page_config(page_title="Global Intelligence Monitor", layout="wide")
st.title("🌍 Global Intelligence Monitor")
st.markdown("Real‑time OSINT from NewsAPI, USGS, OpenSky, Finance, RSS – updated every 30 seconds.")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()
if "last_update" not in st.session_state:
    st.session_state.last_update = None

# Auto-refresh logic
now = datetime.now()
if (st.session_state.last_update is None or 
    (now - st.session_state.last_update).seconds >= REFRESH_INTERVAL):
    with st.spinner("Collecting global intelligence..."):
        new_df = fetch_all_sources()
        if not new_df.empty:
            if st.session_state.data.empty:
                st.session_state.data = new_df
            else:
                st.session_state.data = pd.concat([new_df, st.session_state.data], ignore_index=True)
                # Keep only last MAX_EVENTS
                if len(st.session_state.data) > MAX_EVENTS:
                    st.session_state.data = st.session_state.data.head(MAX_EVENTS)
        st.session_state.last_update = now
        st.rerun()

# Show last update time
st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else 'Never'} | Next refresh in {REFRESH_INTERVAL - (datetime.now() - st.session_state.last_update).seconds if st.session_state.last_update else REFRESH_INTERVAL} seconds.")

# ---- Dashboard ----
if not st.session_state.data.empty:
    df = st.session_state.data.copy()

    # Sidebar filters
    st.sidebar.header("Filters")
    sources = st.sidebar.multiselect("Source", options=df["source"].unique(), default=df["source"].unique())
    event_types = st.sidebar.multiselect("Event Type", options=df["type"].unique(), default=df["type"].unique())
    filtered = df[df["source"].isin(sources) & df["type"].isin(event_types)]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(filtered))
    col2.metric("Unique Sources", filtered["source"].nunique())
    col3.metric("Average Sentiment", f"{filtered['sentiment'].mean():.2f}")

    # Map
    map_df = filtered.dropna(subset=["lat", "lon"])
    if not map_df.empty:
        st.subheader("🗺️ Geolocated Events")
        st.map(map_df[["lat", "lon"]])

    # Table
    st.subheader("📡 Live Intelligence Feed")
    display_cols = ["timestamp", "source", "title", "location", "sentiment", "type"]
    st.dataframe(filtered[display_cols].head(50), use_container_width=True)

    # Sentiment trend
    if len(filtered) > 1:
        trend = filtered[["timestamp_dt", "sentiment"]].dropna().sort_values("timestamp_dt")
        if not trend.empty:
            fig = px.line(trend, x="timestamp_dt", y="sentiment", title="Sentiment Trend Over Time")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for first events... Refresh page in a few seconds.")

# Manual refresh button
if st.button("Refresh Now"):
    st.rerun()
