from __future__ import annotations
import sys, uuid, json, warnings, datetime as dt
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import yfinance as yf
from dateutil import tz
import swisseph as swe
import streamlit as st
import sqlite3
import plotly.graph_objs as go
from sklearn.ensemble import GradientBoostingClassifier
import pytz
from astral import LocationInfo
from astral.sun import sun

warnings.filterwarnings("ignore")

DB_FILE = "astromarketx_gann.db"
if not Path(DB_FILE).exists():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS forecasts (
        id TEXT PRIMARY KEY, ts TEXT, ticker TEXT,
        timeframe TEXT, sentiment TEXT, conf REAL,
        astro_score REAL, tech_score REAL, ml_score REAL,
        risk_score REAL, rationale TEXT
    )''')
    conn.commit()
    conn.close()

def load_yahoo_data(ticker: str, start="2015-01-01"):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if hasattr(df.index, "name") and df.index.name:
        df.reset_index(inplace=True)
    if any(isinstance(col, tuple) for col in df.columns):
        df.columns = [col[1] if isinstance(col, tuple) and len(col)>1 else col for col in df.columns]
    df.columns = [str(col).lower() for col in df.columns]
    if 'close' not in df.columns and 'adj close' in df.columns:
        df['close'] = df['adj close']
    for n in ['open','high','low','close','volume']:
        if n not in df.columns:
            df[n] = np.nan
    return df

def julian_day(date: dt.datetime):
    return swe.julday(date.year, date.month, date.day, date.hour + date.minute/60.0 + date.second/3600.0)

def lahiri_ayanamsa(jd): return swe.get_ayanamsa_ut(jd)
def nirayana(moon_lon, jd): return (moon_lon - lahiri_ayanamsa(jd)) % 360

def get_planets(date: dt.datetime):
    jd = julian_day(date)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    planets = {}
    for code, name in [
        (swe.SUN,"Sun"), (swe.MOON,"Moon"), (swe.MERCURY,"Mercury"),
        (swe.VENUS,"Venus"), (swe.MARS,"Mars"), (swe.JUPITER,"Jupiter"),
        (swe.SATURN,"Saturn"), (swe.URANUS,"Uranus"), (swe.NEPTUNE,"Neptune"),
        (swe.PLUTO,"Pluto"), (swe.MEAN_NODE, "Rahu")
    ]:
        try:
            pos, _speed = swe.calc_ut(jd, code)
            lon = nirayana(pos[0], jd) % 360
            decl = pos[1]
            planets[name] = {"lon": lon, "decl": decl}
        except Exception:
            planets[name] = {"lon": 0, "decl": 0}
    return planets

def calc_nakshatra(moon_lon_deg):
    NAMES = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta",
        "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
        "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
    ]
    idx = int((moon_lon_deg % 360) // (360/27))
    pada = int(((moon_lon_deg % (360/27)) // (360/27/4))) + 1
    return NAMES[idx], idx+1, pada

def vimshottari_dasha(moon_longitude: float, dob: dt.date, eval_date: dt.date):
    sequence = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
    periods = {"Ketu":7, "Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
    degrees = moon_longitude % 360
    constellation = int(degrees // (360/27))
    lord = sequence[constellation % 9]
    residue = (degrees % (360/9)) / (360/9)
    balance = periods[lord] * (1 - residue)
    timeline = []
    start_idx = sequence.index(lord)
    curr = dob
    for i in range(9):
        dasa_lord = sequence[(start_idx+i)%9]
        dasa_years = periods[dasa_lord]
        d_start = curr
        if i == 0:
            d_years = balance
        else:
            d_years = dasa_years
        curr += dt.timedelta(days=int(round(d_years*365.25)))
        timeline.append((dasa_lord, d_start, curr))
        if curr > eval_date: break
    for lord, start, end in timeline:
        if start <= eval_date < end:
            return lord, timeline
    return timeline[-1][0], timeline

def planetary_conjunctions(planets: Dict, orb=3.0):
    pairs = []
    keys = list(planets.keys())
    for i, pl1 in enumerate(keys):
        for j in range(i+1, len(keys)):
            pl2 = keys[j]
            diff = abs((planets[pl1]["lon"] - planets[pl2]["lon"] + 180) % 360 - 180)
            if diff <= orb:
                pairs.append((pl1, pl2, diff))
    return pairs

def gann_time_cycles(date: dt.date, start: dt.date):
    delta_days = (date-start).days
    harmonics = []
    for cyc in [7,28,49,90,144,180,225,240,288,315,360,365,720]:
        if cyc > 0 and delta_days > 0 and delta_days % cyc == 0:
            harmonics.append(f"{cyc}-day Gann cycle active")
    return harmonics

def gann_angle(price: float, start_price: float):
    if price <= 0: return 0
    delta = price - start_price
    angle = np.degrees(np.arctan(abs(delta/(start_price if start_price else 1))))
    return angle

def calc_tech_indicators(df: pd.DataFrame):
    df = df.copy()
    df["rsi"] = df["close"].rolling(14).apply(
        lambda s: 100-100/(1 + s.pct_change().add(1e-5).prod()), raw=False)
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["macdsignal"] = df["macd"].ewm(span=9).mean()
    df["atr"] = (df["high"]-df["low"]).rolling(14).mean()
    df["bbupper"] = df["close"].rolling(20).mean() + 2*df["close"].rolling(20).std()
    df["bblower"] = df["close"].rolling(20).mean() - 2*df["close"].rolling(20).std()
    return df

def calc_tech_score(df: pd.DataFrame):
    latest = df.iloc[-1]
    score = 0
    rationale = []
    if latest.get("rsi", 50) < 30:
        score += 1; rationale.append("RSI<30 (oversold): +1")
    if latest.get("rsi", 50) > 70:
        score -= 1; rationale.append("RSI>70 (overbought): -1")
    if latest["close"] > latest.get("bbupper", latest["close"]+1):
        score -= 1; rationale.append("Price above upper BB: -1")
    if latest["close"] < latest.get("bblower", latest["close"]-1):
        score += 1; rationale.append("Price below lower BB: +1")
    if latest.get("macd", 0) > latest.get("macdsignal", 0):
        score += 1; rationale.append("MACD bullish crossover: +1")
    if latest.get("macd", 0) < latest.get("macdsignal", 0):
        score -= 1; rationale.append("MACD bear crossover: -1")
    score = float(np.tanh(score/4))
    rationale.append(f"Technical score (normalized): {score:+.2f}")
    return score, rationale

class MLModel:
    def __init__(self):
        self.model = None
    def train(self, X, y):
        from numpy import unique
        if len(y)<32 or len(unique(y)) < 2:
            self.model = None
            return
        self.model = GradientBoostingClassifier(n_estimators=100)
        self.model.fit(X, y)
    def predict_prob(self, X_last):
        if self.model is None:
            return 0.5
        return float(self.model.predict_proba([X_last])[0][1])

ml_model = MLModel()
def risk_score(df):
    returns = np.log(df["close"]).diff()
    vol = returns.rolling(20).std().iloc[-1] if not returns.isnull().all() else 0
    maxdd = (df["close"]/df["close"].cummax() - 1).min() if not df["close"].isnull().all() else 0
    agg = float(np.clip((vol or 0) + abs(maxdd or 0), 0, 1))
    return agg
