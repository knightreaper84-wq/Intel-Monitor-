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
    """Fetch news events from NewsAPI"""
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
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        events = []
        for art in articles:
            text = f"{art['title']} {art['description'] or ''}"
            events.append({
                "source": "NewsAPI",
                "title": art.get('title', 'N/A'),
                "description": (art.get('description') or "")[:200],
                "published": art.get('publishedAt', datetime.now().isoformat()),
                "location": extract_location(text),
                "url": art.get('url', ''),
                "raw_text": text,
                "type": "news",
                "severity": 5,
                "lat": None,
                "lon": None,
                "sentiment": 0.0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        return events
    except Exception as e:
        logging.error(f"NewsAPI fetch error: {e}")
        return []

def fetch_usgs() -> List[Dict]:
    """Fetch earthquake events from USGS"""
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_hour.geojson"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [0, 0])
            mag = props.get('mag', 0)
            events.append({
                "source": "USGS",
                "title": f"Earthquake M{mag}",
                "description": f"Magnitude {mag} at {props.get('place', 'Unknown')}",
                "published": datetime.fromtimestamp(props.get("time", 0)/1000).isoformat(),
                "location": props.get('place', 'Unknown'),
                "url": props.get('url', ''),
                "raw_text": f"earthquake magnitude {mag} {props.get('place', '')}",
                "type": "disaster",
                "severity": min(mag, 10),
                "lat": coords[1] if len(coords) > 1 else None,
                "lon": coords[0] if len(coords) > 0 else None,
                "sentiment": -0.3,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        return events
    except Exception as e:
        logging.error(f"USGS fetch error: {e}")
        return []

def fetch_opensky() -> List[Dict]:
    """Fetch aviation data from OpenSky"""
    url = "https://opensky-network.org/api/states/all"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        states = data.get("states", [])[:15]
        events = []
        for s in states:
            if s is None or len(s) < 11 or s[5] is None or s[6] is None:
                continue
            events.append({
                "source": "OpenSky",
                "title": f"Aircraft {s[1] or 'Unknown'} ({s[0] or 'N/A'})",
                "description": f"Altitude: {s[7] or 0}m, Speed: {s[9] or 0}m/s",
                "published": datetime.now().isoformat(),
                "location": f"{s[6]:.2f}, {s[5]:.2f}",
                "url": "https://opensky-network.org",
                "raw_text": "aviation traffic",
                "type": "aviation",
                "severity": 3,
                "lat": s[6],
                "lon": s[5],
                "sentiment": 0.0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        return events
    except Exception as e:
        logging.error(f"OpenSky fetch error: {e}")
        return []

def fetch_economic() -> List[Dict]:
    """Fetch economic data from Yahoo Finance"""
    events = []
    try:
        import yfinance as yf
        tickers = ["^GSPC", "GC=F", "BTC-USD"]
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="1d")
                if not data.empty and len(data) > 0:
                    latest = data["Close"].iloc[-1]
                    prev = data["Close"].iloc[0] if len(data) > 1 else latest
                    change = ((latest - prev) / prev) * 100 if prev != 0 else 0
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
                        "lon": None,
                        "sentiment": 0.1 if change > 0 else -0.1,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
            except Exception as e:
                logging.error(f"yfinance error for {ticker}: {e}")
                continue
    except ImportError:
        logging.warning("yfinance not installed")
    except Exception as e:
        logging.error(f"Economic data fetch error: {e}")
    return events

def fetch_rss() -> List[Dict]:
    """Fetch RSS news feeds"""
    try:
        import feedparser
    except ImportError:
        logging.warning("feedparser not installed")
        return []

    feeds = [
        "http://rss.cnn.com/rss/edition.rss",
        "http://feeds.bbci.co.uk/news/world/rss.xml"
    ]
    events = []
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                text = f"{entry.get('title', '')} {entry.get('summary', '')}"
                events.append({
                    "source": feed_url.split("/")[2],
                    "title": entry.get('title', 'N/A'),
                    "description": (entry.get('summary') or "")[:200],
                    "published": entry.get("published", datetime.now().isoformat()),
                    "location": extract_location(text),
                    "url": entry.get('link', ''),
                    "raw_text": text,
                    "type": "news",
                    "severity": 5,
                    "lat": None,
                    "lon": None,
                    "sentiment": 0.0,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
        except Exception as e:
            logging.error(f"RSS feed error for {feed_url}: {e}")
            continue
    return events

def extract_location(text: str) -> str:
    """Extract location from text using NLP"""
    if nlp is None or not text:
        return "Unknown"
    try:
        doc = nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                return ent.text
    except Exception as e:
        logging.error(f"Location extraction error: {e}")
    return "Unknown"

def geocode_event(event: Dict) -> Dict:
    """Enrich event with geocoding data"""
    if event.get("lat") is not None and event.get("lon") is not None:
        return event
    
    loc_name = event.get("location", "")
    if loc_name and loc_name != "Unknown":
        try:
            loc = geolocator.geocode(loc_name, timeout=3)
            if loc:
                event["lat"] = loc.latitude
                event["lon"] = loc.longitude
        except Exception as e:
            logging.error(f"Geocoding error for {loc_name}: {e}")
    
    return event

def compute_sentiment(text: str) -> float:
    """Compute sentiment score from text"""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return 0.0

# ---- Main collection (FIXED) ----
def fetch_all_sources() -> pd.DataFrame:
    """Fetch and enrich data from all sources"""
    all_events = []
    
    # Collect from all sources
    all_events.extend(fetch_newsapi())
    all_events.extend(fetch_usgs())
    all_events.extend(fetch_opensky())
    all_events.extend(fetch_economic())
    all_events.extend(fetch_rss())

    if not all_events:
        return pd.DataFrame()

    # Deduplicate by title
    seen = set()
    unique = []
    for e in all_events:
        if e["title"] not in seen:
            seen.add(e["title"])
            unique.append(e)

    # Enrich events BEFORE creating DataFrame
    enriched = []
    for e in unique:
        # Compute sentiment if not already set
        if e.get("sentiment") == 0.0:
            e["sentiment"] = compute_sentiment(e.get("raw_text", ""))
        
        # Geocode
        e = geocode_event(e)
        
        enriched.append(e)

    # Create DataFrame from enriched data
    df = pd.DataFrame(enriched)
    
    if not df.empty:
        # Ensure timestamp_dt exists
        df["timestamp_dt"] = pd.to_datetime(df.get("published", datetime.now()), errors="coerce")
        df = df.sort_values("timestamp_dt", ascending=False)
    
    return df

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Global Intelligence Monitor", layout="wide")
st.title("🌍 Global Intelligence Monitor")
st.markdown("Real‑time OSINT from NewsAPI, USGS, OpenSky, Finance, RSS – updated every 30 seconds.")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()
if "last_update" not in st.session_state:
    st.session_state.last_update = None

# Manual refresh button (FIXED - no auto st.rerun loop)
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = None

# Auto-refresh check (without causing infinite loop)
now = datetime.now()
should_refresh = False

if st.session_state.last_update is None:
    should_refresh = True
elif (now - st.session_state.last_update).seconds >= REFRESH_INTERVAL:
    should_refresh = True

if should_refresh:
    with st.spinner("Collecting global intelligence..."):
        new_df = fetch_all_sources()
        if not new_df.empty:
            if st.session_state.data.empty:
                st.session_state.data = new_df
            else:
                st.session_state.data = pd.concat([new_df, st.session_state.data], ignore_index=True)
                # Keep only last MAX_EVENTS
                if len(st.session_state.data) > MAX_EVENTS:
                    st.session_state.data = st.session_state.data.head(MAX_EVENTS).reset_index(drop=True)
        st.session_state.last_update = now

# Show last update time
if st.session_state.last_update:
    seconds_since = (datetime.now() - st.session_state.last_update).seconds
    seconds_until = max(0, REFRESH_INTERVAL - seconds_since)
    st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')} | Next refresh in ~{seconds_until}s")
else:
    st.caption("Waiting for data...")

# ---- Dashboard ----
if not st.session_state.data.empty:
    df = st.session_state.data.copy()

    # Sidebar filters
    st.sidebar.header("Filters")
    if "source" in df.columns:
        sources = st.sidebar.multiselect("Source", options=sorted(df["source"].unique()), default=sorted(df["source"].unique()))
    else:
        sources = []
    
    if "type" in df.columns:
        event_types = st.sidebar.multiselect("Event Type", options=sorted(df["type"].unique()), default=sorted(df["type"].unique()))
    else:
        event_types = []
    
    # Apply filters
    if sources and event_types:
        filtered = df[(df["source"].isin(sources)) & (df["type"].isin(event_types))]
    elif sources:
        filtered = df[df["source"].isin(sources)]
    elif event_types:
        filtered = df[df["type"].isin(event_types)]
    else:
        filtered = df

    if not filtered.empty:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", len(filtered))
        col2.metric("Unique Sources", filtered["source"].nunique() if "source" in filtered.columns else 0)
        avg_sentiment = filtered["sentiment"].mean() if "sentiment" in filtered.columns else 0
        col3.metric("Average Sentiment", f"{avg_sentiment:.2f}")

        # Map
        map_df = filtered.dropna(subset=["lat", "lon"]) if "lat" in filtered.columns and "lon" in filtered.columns else pd.DataFrame()
        if not map_df.empty:
            st.subheader("🗺️ Geolocated Events")
            st.map(map_df[["lat", "lon"]])

        # Table
        st.subheader("📡 Live Intelligence Feed")
        display_cols = [col for col in ["timestamp", "source", "title", "location", "sentiment", "type"] if col in filtered.columns]
        st.dataframe(filtered[display_cols].head(50), use_container_width=True)

        # Sentiment trend
        if "timestamp_dt" in filtered.columns and "sentiment" in filtered.columns:
            trend = filtered[["timestamp_dt", "sentiment"]].dropna().sort_values("timestamp_dt")
            if not trend.empty and len(trend) > 1:
                fig = px.line(trend, x="timestamp_dt", y="sentiment", title="Sentiment Trend Over Time")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events match your filters. Try adjusting them.")
else:
    st.info("⏳ Waiting for first events... This dashboard will auto-refresh every 30 seconds.")
