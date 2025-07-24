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
def lahiri_ayanamsa(jd):
    return swe.get_ayanamsa_ut(jd)
def nirayana(moon_lon, jd):
    return (moon_lon - lahiri_ayanamsa(jd)) % 360
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
def calc_minor_harmonic_aspects(planets: Dict) -> Tuple[float, List[str]]:
    aspect_defs = {
        "conj": (0, 5, 8), "opp": (180, 5, -10), "square": (90, 3, -8), "trine": (120, 3, 5),
        "sextile": (60, 2.5, 5), "semi-sextile": (30, 1.5, 2), "quintile": (72, 1.5, 2),
        "biquintile": (144, 1.5, 1), "semi-square": (45, 1.5, -2), "sesqui": (135, 1.5, -3)
    }
    harmonic_aspects = [51.43, 102.86, 154.29, 205.71, 257.14, 308.57]  # 7th
    minor_score = 0
    rationale = []
    keys = list(planets.keys())
    for i, p1 in enumerate(keys):
        for j, p2 in enumerate(keys):
            if j <= i: continue
            lon1, lon2 = planets[p1]["lon"], planets[p2]["lon"]
            diff = abs((lon1 - lon2 + 180)%360 -180)
            for aname, (adeg, aorb, ascore) in aspect_defs.items():
                if abs(diff-adeg) <= aorb:
                    minor_score += ascore * (1-abs(diff-adeg)/aorb)
                    rationale.append(f"{p1}/{p2} {aname} ({adeg}Â°): {ascore:+}")
            for hdeg in harmonic_aspects:
                if abs(diff-hdeg) <=1:
                    minor_score += 1
                    rationale.append(f"{p1}/{p2} 7th harmonic ({hdeg:.2f}Â°): +1")
    for i, p1 in enumerate(keys):
        for j, p2 in enumerate(keys):
            if j <= i: continue
            d1, d2 = planets[p1]["decl"], planets[p2]["decl"]
            if abs(d1-d2) <= 1:
                minor_score += 2
                rationale.append(f"{p1}/{p2} Parallel (decl. {d1:.2f}Â°/{d2:.2f}Â°): +2")
            elif abs(d1+d2) <= 1:
                minor_score -= 2
                rationale.append(f"{p1}/{p2} Contra-Parallel: -2")
    for i, p1 in enumerate(keys):
        for j, p2 in enumerate(keys):
            if j <= i: continue
            mp = ((planets[p1]["lon"] + planets[p2]["lon"])/2)%360
            for k in keys:
                if k == p1 or k == p2: continue
                diff = abs((planets[k]["lon"]-mp + 180)%360 - 180)
                if diff <= 2:
                    minor_score += 3
                    rationale.append(f"{k} at midpoint({p1}/{p2}) {mp:.2f}Â°: +3")
    rationale.append(f"Accum. minor/harmonic aspect score: {minor_score:+.2f}")
    return np.tanh(minor_score/15), rationale
SECTOR_PLANETS = {
    'Information Technology': ['Mercury', 'Uranus'],
    'Financials': ['Jupiter', 'Venus'],
    'Healthcare': ['Moon', 'Neptune'],
    'Energy': ['Mars', 'Pluto', 'Sun'],
    'Consumer Discretionary': ['Venus', 'Moon'],
    'Industrials': ['Saturn', 'Mars'],
    'Materials': ['Saturn', 'Mars'],
    'Utilities': ['Saturn', 'Sun'],
    'Telecom': ['Mercury', 'Uranus'],
    'Real Estate': ['Venus', 'Saturn'],
}
def calc_sector_scores(planets: Dict, astro_score: float):
    sector_table = []
    for sector, rulers in SECTOR_PLANETS.items():
        sector_rationale = []
        score = 0
        for planet in rulers:
            pl = planets.get(planet)
            if not pl: continue
            lon = pl["lon"]
            if 30 <= lon < 60 or 90 <= lon < 120:
                score += 0.1
                sector_rationale.append(f"{planet} dignified ({lon:.1f}Â°)")
            if 210 <= lon < 240 or 270 <= lon < 300:
                score -= 0.1
                sector_rationale.append(f"{planet} in fall ({lon:.1f}Â°)")
        tot_score = astro_score + score
        tot_score = max(-1, min(1, tot_score))
        if tot_score>0.3: sent = "Bullish"
        elif tot_score<-0.3: sent = "Bearish"
        else: sent = "Neutral"
        sector_table.append({
            "Sector": sector, "AstroScore": tot_score, "Sentiment": sent, "Why": "; ".join(sector_rationale)
        })
    return sector_table
ECLIPSES_2025 = [dt.date(2025, 4, 8), dt.date(2025, 10, 2)]
def get_planet_speed(jd: float, planet_code: int) -> float:
    try: return swe.calc_ut(jd, planet_code)[0][3]
    except: return 0.0
def detect_planetary_stations(date: dt.date) -> Dict[str, str]:
    jd_today = julian_day(dt.datetime.combine(date, dt.time(12, 0)))
    jd_yest = jd_today - 1
    codes = {
        'Mercury': swe.MERCURY,
        'Venus': swe.VENUS,
        'Mars': swe.MARS,
        'Jupiter': swe.JUPITER,
        'Saturn': swe.SATURN,
    }
    stations = {}
    for name, code in codes.items():
        yest = get_planet_speed(jd_yest, code)
        today = get_planet_speed(jd_today, code)
        if yest > 0 and today <= 0:
            stations[name] = 'station_retrograde'
        elif yest < 0 and today >= 0:
            stations[name] = 'station_direct'
        else:
            stations[name] = 'none'
    return stations
def count_retrograde_planets(date: dt.date) -> int:
    jd= julian_day(dt.datetime.combine(date, dt.time(12, 0)))
    return sum(1 for code in [swe.MERCURY, swe.VENUS, swe.MARS, swe.JUPITER, swe.SATURN] if get_planet_speed(jd,code)<0)
def is_near_eclipse(date: dt.date, eclipse_list: List[dt.date], days_window=5):
    near = [e for e in eclipse_list if abs((date-e).days)<=days_window]
    return (len(near)>0, near)
def add_timing_window_signals(forecast_date: dt.date, rationale: List[str]):
    stations = detect_planetary_stations(forecast_date)
    retrograde_count = count_retrograde_planets(forecast_date)
    eclipse_near, eclipses = is_near_eclipse(forecast_date, ECLIPSES_2025)
    for planet,state in stations.items():
        if state=='station_direct': rationale.append(f"{planet} stationary direct today â€“ bullish momentum signal")
        elif state=='station_retrograde': rationale.append(f"{planet} stationary retrograde today â€“ caution, volatility")
    if retrograde_count>=2: rationale.append(f"Retrograde cluster of {retrograde_count}")
    elif retrograde_count==1: rationale.append(f"Single retrograde planet â€“ mild caution")
    if eclipse_near: rationale.append(f"Eclipse window near {', '.join([e.strftime('%b %d') for e in eclipses])}")
    station_score = sum(0.3 if s=='station_direct' else -0.3 if s=='station_retrograde' else 0 for s in stations.values())
    retrograde_score = -0.1*retrograde_count
    eclipse_score = -0.4 if eclipse_near else 0
    tot = station_score + retrograde_score + eclipse_score
    rationale.append(f"Timing window combined score: {tot:+.2f}")
    return tot
def get_lunar_phase(date: dt.date):
    jd= julian_day(dt.datetime.combine(date, dt.time(12, 0)))
    sun_lon = swe.calc_ut(jd, swe.SUN)[0][0]
    moon_lon = swe.calc_ut(jd, swe.MOON)[0][0]
    ayan = swe.get_ayanamsa_ut(jd)
    sun_lon_sid = (sun_lon - ayan)%360
    moon_lon_sid = (moon_lon - ayan)%360
    angle = (moon_lon_sid - sun_lon_sid)%360
    if angle<22.5 or angle>337.5: phase = "New Moon"
    elif 67.5 < angle <=112.5: phase = "First Quarter"
    elif 157.5< angle <=202.5: phase="Full Moon"
    elif 247.5< angle<=292.5: phase="Last Quarter"
    elif angle<180: phase="Waxing"
    else: phase="Waning"
    return phase, angle
def annotate_lunar_phase(date: dt.date, rationale: List[str], sector=None):
    phase, angle = get_lunar_phase(date)
    if phase in ["New Moon","Full Moon","First Quarter","Last Quarter"]:
        rationale.append(f"Lunar phase: {phase} (Sun-Moon: {angle:.1f}Â°) â€“ high impact window")
        if sector in {"Consumer Discretionary","Healthcare","Food"}:
            rationale.append(f"{sector} sector extra sensitive to {phase} phase")
    else:
        rationale.append(f"Lunar phase: {phase} (Sun-Moon: {angle:.1f}Â°)")
    return 0.2 if phase in ["New Moon","Full Moon"] else 0.1 if phase in ["First Quarter","Last Quarter"] else 0
PLANET_ORDER = ['Sun', 'Venus', 'Mercury', 'Moon', 'Saturn', 'Jupiter', 'Mars']
def get_planetary_hour(date_time: dt.datetime, latitude: float, longitude: float, tz_str: str):
    loc = LocationInfo("", "", tz_str, latitude, longitude)
    s = sun(loc.observer, date=date_time.date(), tzinfo=pytz.timezone(tz_str))
    sunrise = s['sunrise']
    sunset = s['sunset']
    is_day = sunrise <= date_time < sunset
    if is_day:
        day_length = (sunset-sunrise).total_seconds()
        hour_length = day_length / 12
        seconds_since_sunrise = (date_time-sunrise).total_seconds()
        hour_index = int(seconds_since_sunrise // hour_length)
    else:
        from astral.sun import sun as nextsun
        next_sunrise = nextsun(loc.observer, date=date_time.date()+dt.timedelta(days=1), tzinfo=pytz.timezone(tz_str))['sunrise']
        night_length = (next_sunrise-sunset).total_seconds()
        seconds_since_sunset = (date_time-sunset).total_seconds() if date_time>sunset else (date_time+dt.timedelta(days=1)-sunset).total_seconds()
        hour_length = night_length / 12
        hour_index = int(seconds_since_sunset // hour_length)
        hour_index += 12
    weekday = date_time.weekday()
    weekday_sun_start = (weekday + 1)%7
    DAY_RULERS = ['Sun','Moon','Mars','Mercury','Jupiter','Venus','Saturn']
    day_ruler = DAY_RULERS[weekday_sun_start]
    day_ruler_index = PLANET_ORDER.index(day_ruler)
    ruler_index = (day_ruler_index + hour_index)%7
    return PLANET_ORDER[ruler_index]
  else:
            stations[name] = 'none'
    return stations
def count_retrograde_planets(date: dt.date) -> int:
    jd= julian_day(dt.datetime.combine(date, dt.time(12, 0)))
    return sum(1 for code in [swe.MERCURY, swe.VENUS, swe.MARS, swe.JUPITER, swe.SATURN] if get_planet_speed(jd,code)<0)
def is_near_eclipse(date: dt.date, eclipse_list: List[dt.date], days_window=5):
    near = [e for e in eclipse_list if abs((date-e).days)<=days_window]
    return (len(near)>0, near)
def add_timing_window_signals(forecast_date: dt.date, rationale: List[str]):
    stations = detect_planetary_stations(forecast_date)
    retrograde_count = count_retrograde_planets(forecast_date)
    eclipse_near, eclipses = is_near_eclipse(forecast_date, ECLIPSES_2025)
    for planet,state in stations.items():
        if state=='station_direct': rationale.append(f"{planet} stationary direct today â€“ bullish momentum signal")
        elif state=='station_retrograde': rationale.append(f"{planet} stationary retrograde today â€“ caution, volatility")
    if retrograde_count>=2: rationale.append(f"Retrograde cluster of {retrograde_count}")
    elif retrograde_count==1: rationale.append(f"Single retrograde planet â€“ mild caution")
    if eclipse_near: rationale.append(f"Eclipse window near {', '.join([e.strftime('%b %d') for e in eclipses])}")
    station_score = sum(0.3 if s=='station_direct' else -0.3 if s=='station_retrograde' else 0 for s in stations.values())
    retrograde_score = -0.1*retrograde_count
    eclipse_score = -0.4 if eclipse_near else 0
    tot = station_score + retrograde_score + eclipse_score
    rationale.append(f"Timing window combined score: {tot:+.2f}")
    return tot
def get_lunar_phase(date: dt.date):
    jd= julian_day(dt.datetime.combine(date, dt.time(12, 0)))
    sun_lon = swe.calc_ut(jd, swe.SUN)[0][0]
    moon_lon = swe.calc_ut(jd, swe.MOON)[0][0]
    ayan = swe.get_ayanamsa_ut(jd)
    sun_lon_sid = (sun_lon - ayan)%360
    moon_lon_sid = (moon_lon - ayan)%360
    angle = (moon_lon_sid - sun_lon_sid)%360
    if angle<22.5 or angle>337.5: phase = "New Moon"
    elif 67.5 < angle <=112.5: phase = "First Quarter"
    elif 157.5< angle <=202.5: phase="Full Moon"
    elif 247.5< angle<=292.5: phase="Last Quarter"
    elif angle<180: phase="Waxing"
    else: phase="Waning"
    return phase, angle
def annotate_lunar_phase(date: dt.date, rationale: List[str], sector=None):
    phase, angle = get_lunar_phase(date)
    if phase in ["New Moon","Full Moon","First Quarter","Last Quarter"]:
        rationale.append(f"Lunar phase: {phase} (Sun-Moon: {angle:.1f}Â°) â€“ high impact window")
        if sector in {"Consumer Discretionary","Healthcare","Food"}:
            rationale.append(f"{sector} sector extra sensitive to {phase} phase")
    else:
        rationale
