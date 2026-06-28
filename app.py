import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
import warnings, json, os, uuid, calendar as cal_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="STATIUM · Sports Intelligence",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System — STATIUM Brand Manual ─────────────────────
BRAND_GREEN   = "#00A86B"   # Signal Green
BRAND_DARK    = "#0A0D12"   # Carbon Black
BRAND_GRAPHITE= "#171B22"   # Deep Graphite
BRAND_STEEL   = "#3A404A"   # Steel Gray
BRAND_BLUE    = "#1a6fa8"   # Data Blue (away / secondary)
BRAND_AMBER   = "#d97706"   # Warning / low confidence

# Logo SVG — STATIUM icon: stadium ring + brand triangle
LOGO_SVG = """
<svg width="52" height="52" viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg">
  <rect width="52" height="52" rx="13" fill="#0A0D12"/>
  <ellipse cx="26" cy="22" rx="15" ry="7.5" fill="none" stroke="#00A86B" stroke-width="2"/>
  <line x1="11" y1="22" x2="11" y2="30" stroke="#00A86B" stroke-width="2" stroke-linecap="round"/>
  <line x1="41" y1="22" x2="41" y2="30" stroke="#00A86B" stroke-width="2" stroke-linecap="round"/>
  <path d="M11,30 Q11,41 26,41 Q41,41 41,30" fill="none" stroke="#00A86B" stroke-width="2"/>
  <ellipse cx="26" cy="22" rx="7" ry="3.5" fill="none" stroke="#00A86B" stroke-width="1" opacity="0.45"/>
  <polygon points="26,8 22.5,15 29.5,15" fill="#00A86B"/>
</svg>
"""

import base64
def logo_img(size=52):
    b64 = base64.b64encode(LOGO_SVG.strip().encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" style="border-radius:10px">'

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  /* ── Reset & Base ── */
  .block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
  html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif !important; }}
  * {{ scroll-behavior: smooth; }}

  /* ── Global lively transitions ── */
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(0,168,107,0.0); }}
    50%      {{ box-shadow: 0 0 0 6px rgba(0,168,107,0.08); }}
  }}
  @keyframes shimmer {{
    0%   {{ background-position: -200px 0; }}
    100% {{ background-position: calc(200px + 100%) 0; }}
  }}

  .main .block-container {{ animation: fadeIn 0.5s ease-out both; }}

  /* Buttons — smooth lift + color sweep */
  .stButton > button {{
    transition: transform 0.18s cubic-bezier(.2,.8,.2,1), box-shadow 0.18s ease, background 0.25s ease, border-color 0.25s ease !important;
  }}
  .stButton > button:hover {{
    transform: translateY(-1px) scale(1.015);
    box-shadow: 0 6px 18px rgba(0,168,107,0.18) !important;
    border-color: {BRAND_GREEN} !important;
    color: {BRAND_GREEN} !important;
  }}
  .stButton > button:active {{ transform: translateY(0) scale(0.98); }}

  /* Download button glow */
  .stDownloadButton > button {{ transition: all 0.2s ease !important; }}
  .stDownloadButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,168,107,0.20) !important; }}

  /* Tabs — underline slide + color fade */
  [data-testid="stTabs"] button {{
    font-weight:600; font-family:'Space Grotesk',sans-serif !important;
    transition: color 0.2s ease, background 0.2s ease !important;
    border-radius: 8px 8px 0 0 !important;
  }}
  [data-testid="stTabs"] button:hover {{
    color: {BRAND_GREEN} !important;
    background: rgba(0,168,107,0.06) !important;
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
    transition: all 0.25s cubic-bezier(.2,.8,.2,1) !important;
  }}
  [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    transition: left 0.3s cubic-bezier(.2,.8,.2,1), width 0.3s cubic-bezier(.2,.8,.2,1) !important;
    background-color: {BRAND_GREEN} !important;
  }}
  [data-testid="stTabs"] [data-baseweb="tab-panel"] {{ animation: fadeIn 0.35s ease-out both; }}

  /* Selectbox / sliders / inputs — soft focus glow */
  [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
  }}
  [data-baseweb="select"] > div:hover, .stTextInput input:hover {{
    border-color: {BRAND_GREEN} !important;
  }}
  [data-baseweb="select"] > div:focus-within, .stTextInput input:focus {{
    box-shadow: 0 0 0 3px rgba(0,168,107,0.15) !important;
    border-color: {BRAND_GREEN} !important;
  }}
  div[data-testid="stSlider"] [role="slider"] {{
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
  }}
  div[data-testid="stSlider"] [role="slider"]:hover {{
    transform: scale(1.25);
    box-shadow: 0 0 0 8px rgba(0,168,107,0.12) !important;
  }}

  /* Expanders — smooth open */
  [data-testid="stExpander"] {{
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    border-radius: 10px !important;
  }}
  [data-testid="stExpander"]:hover {{
    border-color: rgba(0,168,107,0.35) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
  }}
  [data-testid="stExpanderDetails"] {{ animation: fadeIn 0.3s ease-out both; }}

  /* Metric cards pop-in */
  div[data-testid="stMetric"] {{ transition: transform 0.18s ease; }}
  div[data-testid="stMetric"]:hover {{ transform: translateY(-2px); }}

  /* Dataframes — subtle entrance */
  [data-testid="stDataFrame"] {{ animation: fadeIn 0.4s ease-out both; border-radius: 10px; overflow: hidden; }}

  /* Alerts / info boxes — slide in */
  [data-testid="stAlert"] {{
    animation: fadeIn 0.35s ease-out both;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    border-radius: 10px !important;
  }}
  [data-testid="stAlert"]:hover {{ transform: translateX(2px); }}

  /* Sidebar nav buttons (calendar arrows etc.) */
  [data-testid="stSidebar"] .stButton > button {{
    transition: all 0.18s ease !important;
  }}
  [data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(0,168,107,0.10) !important;
  }}

  /* Links */
  a {{ transition: color 0.15s ease, opacity 0.15s ease; }}
  a:hover {{ opacity: 0.75; }}

  /* ── Header ── */
  .stat-header {{
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 8px 0;
    animation: fadeIn 0.6s ease-out both;
  }}
  .stat-logo {{ flex-shrink: 0; }}
  .stat-title {{
    font-size: 2.2rem; font-weight: 800; letter-spacing: 2px;
    color: {BRAND_DARK};
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1.1;
  }}
  .stat-tagline {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem; font-weight: 500; letter-spacing: 1.5px;
    color: {BRAND_GREEN}; text-transform: uppercase; margin-top: 3px;
  }}
  .stat-subtitle {{ color: #64748b; font-size: 0.82rem; margin-top: 1px; }}

  /* ── Gradient separator ── */
  .grad-line {{
    height: 2px; border-radius: 2px;
    background: linear-gradient(90deg, {BRAND_GREEN}, {BRAND_DARK}, transparent);
    margin: 10px 0 18px 0;
  }}

  /* ── Stat cards (top metrics) ── */
  .stat-card {{
    background: white; border-radius: 14px; padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 3px solid {BRAND_GREEN};
    text-align: center;
    transition: transform 0.2s cubic-bezier(.2,.8,.2,1), box-shadow 0.2s ease;
    animation: fadeIn 0.45s ease-out both;
  }}
  .stat-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(0,168,107,0.16);
  }}
  .stat-card-num  {{ font-size: 1.8rem; font-weight: 800; color: {BRAND_DARK}; font-family: 'IBM Plex Mono', monospace; }}
  .stat-card-label{{ font-size: 0.72rem; color: #64748b; margin-top: 2px; letter-spacing: .3px; }}

  /* ── Value Bet Cards ── */
  @keyframes slideInUp {{
    from {{ opacity: 0; transform: translateY(22px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  .vb-card {{
    background: white; border-radius: 16px; padding: 20px 22px;
    margin-bottom: 14px; position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
    transition: transform 0.15s, box-shadow 0.15s;
    animation: slideInUp 0.45s ease-out both;
  }}
  .vb-card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 32px rgba(0,0,0,0.10); }}
  .vb-card-high   {{ border-left: 4px solid {BRAND_GREEN}; box-shadow: 0 4px 20px rgba(0,168,107,0.10); }}
  .vb-card-medium {{ border-left: 4px solid {BRAND_STEEL}; box-shadow: 0 4px 20px rgba(58,64,74,0.08); }}
  .vb-card-low    {{ border-left: 4px solid {BRAND_AMBER}; box-shadow: 0 4px 20px rgba(217,119,6,0.08); }}

  .vb-match   {{ font-size: 1.05rem; font-weight: 700; color: {BRAND_DARK}; margin: 0; font-family: 'Space Grotesk', sans-serif; }}
  .vb-meta    {{ font-size: 0.77rem; color: #64748b; margin-top: 3px; font-family: 'IBM Plex Mono', monospace; }}
  .vb-ctx-row {{ display: flex; gap: 6px; margin: 8px 0; flex-wrap: wrap; }}

  /* ── EV Badge — premium, not casino ── */
  .ev-pill {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 1.15rem; font-weight: 700;
    padding: 5px 14px; border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: .5px;
  }}
  .ev-high   {{ background: rgba(0,168,107,0.10); color: {BRAND_GREEN}; border: 1px solid rgba(0,168,107,0.28); }}
  .ev-medium {{ background: rgba(58,64,74,0.07);  color: {BRAND_STEEL}; border: 1px solid rgba(58,64,74,0.22); }}
  .ev-low    {{ background: rgba(217,119,6,0.09); color: {BRAND_AMBER}; border: 1px solid rgba(217,119,6,0.25); }}

  .conf-tag {{
    font-size: 0.70rem; font-weight: 600; padding: 2px 9px;
    border-radius: 6px; display: inline-block;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: .3px;
  }}
  .conf-high   {{ background: rgba(0,168,107,0.09); color: {BRAND_GREEN}; border: 1px solid rgba(0,168,107,0.25); }}
  .conf-medium {{ background: rgba(58,64,74,0.07);  color: {BRAND_STEEL}; border: 1px solid rgba(58,64,74,0.20); }}
  .conf-low    {{ background: rgba(217,119,6,0.09); color: {BRAND_AMBER}; border: 1px solid rgba(217,119,6,0.22); }}

  .vb-details {{
    font-size: 0.82rem; color: #475569; margin-top: 10px;
    display: flex; flex-wrap: wrap; gap: 14px;
  }}
  .vb-detail-item {{ display: flex; flex-direction: column; }}
  .vb-detail-label {{ font-size: 0.63rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .6px; font-family: 'IBM Plex Mono', monospace; }}
  .vb-detail-val   {{ font-weight: 600; color: {BRAND_DARK}; font-size: 0.90rem; font-family: 'IBM Plex Mono', monospace; }}
  .vb-detail-val.green {{ color: {BRAND_GREEN}; }}
  .vb-detail-val.blue  {{ color: {BRAND_STEEL}; }}

  /* ── Context badges ── */
  .ctx-badge  {{ display:inline-block; font-size:.70rem; font-weight:600; padding:2px 9px; border-radius:6px; font-family:'Space Grotesk',sans-serif; }}
  .ctx-title      {{ background:#fef9c3; color:#854d0e; border:1px solid #fde047; }}
  .ctx-champion   {{ background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }}
  .ctx-champ-won  {{ background:rgba(0,168,107,0.10); color:{BRAND_GREEN}; border:1px solid rgba(0,168,107,0.28); }}
  .ctx-europa     {{ background:rgba(0,168,107,0.08); color:{BRAND_GREEN}; border:1px solid rgba(0,168,107,0.22); }}
  .ctx-mid        {{ background:#f8fafc; color:#64748b; border:1px solid #e2e8f0; }}
  .ctx-dead       {{ background:#f8fafc; color:#94a3b8; border:1px solid #e2e8f0; font-style:italic; }}
  .ctx-nearrel    {{ background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; }}
  .ctx-relegation {{ background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }}

  /* ── Context alert ── */
  .ctx-alert {{
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px;
    padding: 6px 12px; font-size: .78rem; color: #92400e; margin-top: 8px;
    font-family: 'Space Grotesk', sans-serif;
  }}

  /* ── Compact prob bar (inside card) ── */
  .card-prob-bar {{
    display:flex; height:20px; border-radius:4px; overflow:hidden; gap:1px;
    margin: 8px 0 4px;
  }}
  .cpb-home {{ background:{BRAND_GREEN}; display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:600; min-width:28px; font-family:'IBM Plex Mono',monospace; }}
  .cpb-draw {{ background:{BRAND_STEEL}; display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:600; min-width:28px; font-family:'IBM Plex Mono',monospace; }}
  .cpb-away {{ background:{BRAND_BLUE};  display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:600; min-width:28px; font-family:'IBM Plex Mono',monospace; }}

  /* ── Full probability bars (Tab 2) ── */
  .prob-bar-wrap {{ margin: 14px 0 4px; }}
  .prob-bar-label {{ font-size: .68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .6px; margin-bottom: 5px; font-family:'IBM Plex Mono',monospace; }}
  .prob-bar-1x2 {{ display:flex; height:26px; border-radius:6px; overflow:hidden; gap:1px; }}
  .pb-home {{ background:{BRAND_GREEN}; display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:600; min-width:30px; font-family:'IBM Plex Mono',monospace; }}
  .pb-draw {{ background:{BRAND_STEEL}; display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:600; min-width:30px; font-family:'IBM Plex Mono',monospace; }}
  .pb-away {{ background:{BRAND_BLUE};  display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:600; min-width:30px; font-family:'IBM Plex Mono',monospace; }}

  .prob-pills {{ display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }}
  .prob-pill {{
    display:flex; align-items:center; gap:5px; padding:3px 10px;
    background:#f8fafc; border-radius:6px; font-size:.76rem;
    border:1px solid #e2e8f0; font-family:'IBM Plex Mono',monospace;
  }}
  .prob-pill-dot {{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }}
  .pp-green  {{ background:{BRAND_GREEN}; }}
  .pp-blue   {{ background:{BRAND_STEEL}; }}
  .pp-orange {{ background:#f97316; }}
  .pp-purple {{ background:{BRAND_BLUE}; }}
  .prob-pill-val {{ font-weight:600; color:{BRAND_DARK}; }}

  /* ── Form badges ── */
  .form-badge {{ display:inline-block; width:24px; height:24px; border-radius:4px; text-align:center; line-height:24px; font-size:11px; font-weight:700; color:white; margin:1px; font-family:'Space Grotesk',sans-serif; }}
  .fb-w {{ background:{BRAND_GREEN}; }}
  .fb-d {{ background:{BRAND_STEEL}; }}
  .fb-l {{ background:#ef4444; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{ background: white !important; }}

  /* ── Tabs ── */
  [data-testid="stTabs"] button {{ font-weight:600; font-family:'Space Grotesk',sans-serif !important; }}

  /* ── Override Streamlit metric ── */
  div[data-testid="stMetricValue"] {{ font-size:1.5rem !important; font-weight:700 !important; font-family:'IBM Plex Mono',monospace !important; }}
  div[data-testid="stMetricLabel"]  {{ font-size:.72rem !important; font-family:'Space Grotesk',sans-serif !important; }}

  .footer {{
    text-align:center; color:#94a3b8; font-size:.73rem;
    margin-top:2.5rem; padding-top:1rem;
    border-top: 1px solid #e2e8f0;
  }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
FD_BASE   = "https://api.football-data.org/v4"
ODDS_BASE = "https://api.the-odds-api.com/v4"
FDCO_BASE = "https://www.football-data.co.uk/mmz4281"

# Códigos de football-data.co.uk por liga (fd_code → fdco_code)
FDCO_LEAGUES = {
    "PL":  "E0",   # Premier League
    "PD":  "SP1",  # La Liga
    "SA":  "I1",   # Serie A
    "BL1": "D1",   # Bundesliga
    "FL1": "F1",   # Ligue 1
}

# ── Hora local del usuario: Perú (America/Lima) — UTC-5 todo el año, sin DST ──
PERU_OFFSET = timedelta(hours=-5)

def to_lima(dt):
    """Convierte un datetime (con o sin tz, asumido UTC) a hora de Perú (UTC-5)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt + PERU_OFFSET

def fmt_match_dt(date_str, fmt="%a %d/%m · %H:%M"):
    """Formatea una fecha ISO en UTC (ej. '2026-06-11T19:00:00Z') a hora de Perú legible."""
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
    except Exception:
        return str(date_str)
    return to_lima(dt).strftime(fmt) + " (PE)"

LEAGUES = {
    "🌍 Mundial 2026":     {"fd":"WC",  "odds":"soccer_fifa_world_cup",     "games":3, "teams":4, "cl":2,"euro":0,"rel":1,"is_tournament":True,"featured":True},
    "🇬🇧 Premier League": {"fd":"PL",  "odds":"soccer_epl",               "games":38,"teams":20,"cl":4,"euro":6,"rel":3,"off_season":True},
    "🇪🇸 La Liga":         {"fd":"PD",  "odds":"soccer_spain_la_liga",      "games":38,"teams":20,"cl":4,"euro":7,"rel":3,"off_season":True},
    "🇮🇹 Serie A":         {"fd":"SA",  "odds":"soccer_italy_serie_a",      "games":38,"teams":20,"cl":4,"euro":7,"rel":3,"off_season":True},
    "🇩🇪 Bundesliga":      {"fd":"BL1", "odds":"soccer_germany_bundesliga", "games":34,"teams":18,"cl":4,"euro":6,"rel":2,"off_season":True},
    "🇫🇷 Ligue 1":         {"fd":"FL1", "odds":"soccer_france_ligue_one",   "games":34,"teams":18,"cl":3,"euro":5,"rel":3,"off_season":True},
}

MIN_ODDS     = 1.35
MAX_ODDS     = 7.00
MAX_EDGE     = 0.17
SHRINKAGE_K  = 10
DECAY_RATE   = 0.010       # Ligas de club: peso ~0% a 1.5 años
INTL_DECAY   = 0.0006      # Selecciones: peso ~64% a 2 años, ~41% a 3 años
LATE_SEASON  = 5

@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# ── Fuentes internacionales para el modelo WC ─────────────────
# Pesos de competición: eliminatorias = 1.0, torneos mayores ≈ 0.90,
# Nations League ≈ 0.80, amistosos oficiales FIFA ≈ 0.50
# Los amistosos NO oficiales / sub-23 no aparecen en football-data.org.
# Solo se consideran partidos de selecciones absolutas (A-team).
INTL_SOURCES = [
    # (fd_code,  peso_competicion,  etiqueta)
    ("WC",  1.00, "Copa del Mundo"),         # partidos del propio torneo
    ("WCQ", 1.00, "Eliminatorias"),          # clasificatorias WC (requiere plan API ≥ Tier 2)
    ("EC",  0.90, "Eurocopa"),               # Euros para selecciones UEFA
    ("ECQ", 0.85, "Clasif. Eurocopa"),       # clasificatorias UEFA
    ("CA",  0.90, "Copa América"),           # CONMEBOL / CONCACAF
    ("GC",  0.75, "Copa Oro"),               # CONCACAF — clave para México/USA/Canadá
    ("NL",  0.80, "Nations League UEFA"),    # competitivo, fechas FIFA regulares
    ("CNL", 0.75, "Nations League CONCACAF"),# CONCACAF Nations League
    ("WCF", 0.50, "Amistosos FIFA"),         # amistosos oficiales A-team
]

# Países sede 2026: no clasificaron vía eliminatorias →
# sus amistosos / Copa Oro / CONCACAF NL tienen peso elevado (= clasificatorias)
HOST_NATIONS = {"united states", "usa", "u.s.a.", "canada", "mexico", "méxico"}

# ═══════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════
def _fd_get(api_key, endpoint, params=None):
    try:
        r = requests.get(f"{FD_BASE}{endpoint}", headers={"X-Auth-Token": api_key},
                         params=params, timeout=12)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def _odds_get(api_key, endpoint, params=None):
    try:
        r = requests.get(f"{ODDS_BASE}{endpoint}",
                         params={"apiKey": api_key, **(params or {})}, timeout=12)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_season_matches(fd_key, fd_code):
    data = _fd_get(fd_key, f"/competitions/{fd_code}/matches", {"status":"FINISHED"})
    if not data: return pd.DataFrame()
    rows = []
    for m in data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        if ft.get("home") is not None:
            rows.append({"date":m["utcDate"],
                "home_id":m["homeTeam"]["id"],"home_name":m["homeTeam"]["name"],
                "away_id":m["awayTeam"]["id"],"away_name":m["awayTeam"]["name"],
                "home_goals":ft["home"],"away_goals":ft["away"]})
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_upcoming_matches(fd_key, fd_code, days=7):
    today  = datetime.utcnow().strftime("%Y-%m-%d")
    future = (datetime.utcnow()+timedelta(days=days)).strftime("%Y-%m-%d")
    data = _fd_get(fd_key, f"/competitions/{fd_code}/matches",
                   {"status":"SCHEDULED","dateFrom":today,"dateTo":future})
    if not data: return []
    # Nota: en torneos como el Mundial, algunos cruces aún no definen equipo
    # (ej. "Ganador Grupo A") y football-data.org devuelve `name: null` — lo
    # normalizamos a un texto legible para evitar errores de tipo más adelante.
    return [{"id":m["id"],"date":m["utcDate"],
             "home_id":m["homeTeam"]["id"],"home_name":m["homeTeam"].get("name") or "Por definir",
             "away_id":m["awayTeam"]["id"],"away_name":m["awayTeam"].get("name") or "Por definir",
             "matchday":m.get("matchday") or 0} for m in data.get("matches",[])]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_standings(fd_key, fd_code):
    data = _fd_get(fd_key, f"/competitions/{fd_code}/standings")
    if not data: return pd.DataFrame()
    rr_rows = []
    for s in data.get("standings", []):
        stype = s.get("type", "")
        if stype == "TOTAL":
            # Liga — tabla única
            return pd.DataFrame([{
                "position": e["position"], "team_id": e["team"]["id"],
                "team_name": e["team"]["name"], "played": e["playedGames"],
                "points": e["points"], "gf": e["goalsFor"],
                "ga": e["goalsAgainst"], "gd": e["goalDifference"],
            } for e in s.get("table", [])])
        elif stype == "ROUND_ROBIN":
            # Torneo (Mundial, Euros) — grupos separados; los concatenamos
            group_name = s.get("group", "")
            for e in s.get("table", []):
                rr_rows.append({
                    "position": e["position"], "team_id": e["team"]["id"],
                    "team_name": e["team"]["name"], "played": e["playedGames"],
                    "points": e["points"], "gf": e["goalsFor"],
                    "ga": e["goalsAgainst"], "gd": e["goalDifference"],
                    "group": group_name,
                })
    if rr_rows:
        return pd.DataFrame(rr_rows)
    return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_seasons(fd_code, n_seasons=2):
    """
    Descarga hasta n_seasons temporadas históricas desde football-data.co.uk (gratis).
    Devuelve DataFrame con columnas de goles + corners + tiros para uso futuro.
    """
    league_code = FDCO_LEAGUES.get(fd_code)
    if not league_code:
        return pd.DataFrame()

    now = datetime.utcnow()
    sy_current = now.year if now.month >= 7 else now.year - 1

    all_parts = []
    for i in range(1, n_seasons + 1):
        sy = sy_current - i
        ey = sy + 1
        season = f"{str(sy)[-2:]}{str(ey)[-2:]}"
        url = f"{FDCO_BASE}/{season}/{league_code}.csv"
        try:
            df = pd.read_csv(url)
            needed = {"FTHG", "FTAG", "HomeTeam", "AwayTeam", "Date"}
            if not needed.issubset(df.columns):
                continue
            df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam", "Date"])
            df["home_goals"]    = df["FTHG"].astype(int)
            df["away_goals"]    = df["FTAG"].astype(int)
            df["home_name_raw"] = df["HomeTeam"].astype(str).str.strip()
            df["away_name_raw"] = df["AwayTeam"].astype(str).str.strip()
            df["date"]          = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%dT12:00:00Z")
            df = df.dropna(subset=["date"])
            for col, key in [("home_corners","HC"),("away_corners","AC"),
                              ("home_shots","HS"),("away_shots","AS"),
                              ("home_shots_ot","HST"),("away_shots_ot","AST")]:
                df[col] = pd.to_numeric(df[key], errors="coerce") if key in df.columns else np.nan
            keep = ["date","home_name_raw","away_name_raw","home_goals","away_goals",
                    "home_corners","away_corners","home_shots","away_shots","home_shots_ot","away_shots_ot"]
            all_parts.append(df[keep])
        except Exception:
            continue

    if not all_parts:
        return pd.DataFrame()
    return pd.concat(all_parts, ignore_index=True)

def enrich_with_history(season_df, hist_df):
    """
    Añade partidos históricos al DataFrame de la temporada actual.
    Hace fuzzy match de nombres de equipos del CSV histórico contra los IDs del API.
    """
    if hist_df.empty or season_df.empty:
        return season_df

    name_map = {}
    for _, row in season_df.iterrows():
        name_map[row["home_name"].lower().strip()] = (row["home_id"], row["home_name"])
        name_map[row["away_name"].lower().strip()] = (row["away_id"], row["away_name"])

    def lookup(raw):
        q = raw.lower().strip()
        if q in name_map:
            return name_map[q]
        best_s, best_k = 0, None
        for key in name_map:
            s = _sim(q, key)
            if s > best_s:
                best_s, best_k = s, key
        return name_map[best_k] if best_s >= 0.65 else (None, None)

    rows = []
    stat_rows = []
    extra_cols = ["home_corners","away_corners","home_shots","away_shots","home_shots_ot","away_shots_ot"]
    for _, row in hist_df.iterrows():
        h_id, h_name = lookup(row["home_name_raw"])
        a_id, a_name = lookup(row["away_name_raw"])
        if h_id is None or a_id is None:
            continue
        base = {
            "date": row["date"],
            "home_id": h_id, "home_name": h_name,
            "away_id": a_id, "away_name": a_name,
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
        }
        rows.append(base)
        stat_row = {**base}
        for col in extra_cols:
            stat_row[col] = row.get(col, np.nan)
        stat_rows.append(stat_row)

    if not rows:
        return season_df, pd.DataFrame()

    combined = pd.concat([season_df, pd.DataFrame(rows)], ignore_index=True)
    combined.drop_duplicates(subset=["date","home_id","away_id"], keep="first", inplace=True)
    combined.sort_values("date", ascending=False, inplace=True)
    hist_mapped = pd.DataFrame(stat_rows)
    return combined.reset_index(drop=True), hist_mapped

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_intl_matches(fd_key):
    """
    Combina partidos de múltiples competiciones internacionales para construir
    ratings de selecciones. Cada partido lleva:
      - comp_weight : peso competitivo (1.0 = eliminatoria, 0.5 = amistoso)
      - comp_code   : código de competición de origen
      - comp_label  : etiqueta legible
    Los países sede (USA / Canadá / México) reciben peso 1.0 en sus partidos
    de GC / CNL / WCF porque no tuvieron eliminatorias.
    """
    def _is_host(name):
        return any(h in name.lower() for h in HOST_NATIONS)

    def _fetch_one(fd_code, base_w, label):
        data = _fd_get(fd_key, f"/competitions/{fd_code}/matches", {"status": "FINISHED"})
        if not data:
            return []
        rows = []
        for m in data.get("matches", []):
            ft = m.get("score", {}).get("fullTime", {})
            if ft.get("home") is None:
                continue
            home_name = m["homeTeam"]["name"]
            away_name = m["awayTeam"]["name"]
            if fd_code in ("GC", "CNL", "WCF", "CA", "NL") and (
                _is_host(home_name) or _is_host(away_name)
            ):
                comp_w = 1.0
            else:
                comp_w = base_w
            rows.append({
                "_mid":       m.get("id"),
                "date":       m["utcDate"],
                "home_id":    m["homeTeam"]["id"],
                "home_name":  home_name,
                "away_id":    m["awayTeam"]["id"],
                "away_name":  away_name,
                "home_goals": int(ft["home"]),
                "away_goals": int(ft["away"]),
                "comp_weight": comp_w,
                "comp_code":   fd_code,
                "comp_label":  label,
            })
        return rows

    all_rows = []
    seen_ids = set()
    with ThreadPoolExecutor(max_workers=len(INTL_SOURCES)) as ex:
        futures = {ex.submit(_fetch_one, c, w, l): c for c, w, l in INTL_SOURCES}
        for fut in as_completed(futures):
            for row in (fut.result() or []):
                mid = row.pop("_mid")
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.sort_values("date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def upcoming_from_odds(odds_list, season_df, days_ahead=60):
    """
    Para torneos sin fixtures en football-data.org todavía, construye la lista
    de próximos partidos directamente desde los datos de cuotas.
    Hace fuzzy-match de nombres de equipos contra el historial de partidos
    para recuperar los IDs necesarios para el modelo.
    """
    if not odds_list or season_df.empty:
        return []

    # Mapa nombre → id desde el historial de selecciones
    name_to_id: dict = {}
    for _, row in season_df.iterrows():
        name_to_id[row["home_name"].lower()] = (row["home_name"], row["home_id"])
        name_to_id[row["away_name"].lower()] = (row["away_name"], row["away_id"])

    def fuzzy_lookup(query):
        """Devuelve (canonical_name, team_id) o (query, None) si no hay match."""
        q = query.lower().strip()
        if q in name_to_id:
            return name_to_id[q]
        qn = _norm_country(q)
        if qn in name_to_id:
            return name_to_id[qn]
        # Fuzzy sobre todas las claves (con normalización de alias de selecciones)
        best_score, best_key = 0, None
        for key in name_to_id:
            s = max(SequenceMatcher(None, q, key).ratio(),
                    SequenceMatcher(None, qn, _norm_country(key)).ratio())
            if s > best_score:
                best_score, best_key = s, key
        if best_score >= 0.68:
            return name_to_id[best_key]
        return query, None   # sin match — mostramos pero sin modelo

    now_dt    = datetime.utcnow().replace(tzinfo=None)
    cutoff_dt = now_dt + timedelta(days=days_ahead)

    upcoming = []
    seen = set()
    for om in odds_list:
        ct = om.get("commence_time", "")
        if not ct:
            continue
        # Comparación robusta usando objetos datetime (no strings con "Z")
        try:
            ct_dt = datetime.fromisoformat(ct.replace("Z", "")).replace(tzinfo=None)
        except Exception:
            continue
        if ct_dt < now_dt or ct_dt > cutoff_dt:
            continue
        home_raw = om.get("home_team") or ""
        away_raw = om.get("away_team") or ""
        if not home_raw or not away_raw:
            continue
        # Dedup por (fecha_día, local, visitante) — permite misma pareja en distintos días
        day_key = (ct_dt.date().isoformat(), home_raw.lower(), away_raw.lower())
        if day_key in seen:
            continue
        seen.add(day_key)

        h_name, h_id = fuzzy_lookup(home_raw)
        a_name, a_id = fuzzy_lookup(away_raw)

        upcoming.append({
            "id":        om.get("id", ""),
            "date":      ct if ct.endswith("Z") else ct + "Z",
            "home_id":   h_id if h_id is not None else -hash(home_raw),
            "home_name": h_name,
            "away_id":   a_id if a_id is not None else -hash(away_raw),
            "away_name": a_name,
            "matchday":  0,
            "_from_odds": True,   # bandera interna
            "_h_matched": h_id is not None,
            "_a_matched": a_id is not None,
        })

    upcoming.sort(key=lambda x: x["date"])
    return upcoming

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_odds(odds_key, sport_key):
    data = _odds_get(odds_key, f"/sports/{sport_key}/odds/",
                     {"regions":"eu","markets":"h2h,totals,btts,bookie_corners","oddsFormat":"decimal"})
    return data if isinstance(data, list) else []

# ═══════════════════════════════════════════════════════════
# CONTEXTO COMPETITIVO
# ═══════════════════════════════════════════════════════════
def is_title_decided(standings_df, remaining):
    if standings_df.empty or len(standings_df) < 2: return False
    if "group" in standings_df.columns: return False   # torneo: nunca "título decidido"
    pts1 = standings_df[standings_df["position"]==1]["points"].values
    pts2 = standings_df[standings_df["position"]==2]["points"].values
    if len(pts1)==0 or len(pts2)==0: return False
    return int(pts1[0]) > int(pts2[0]) + remaining * 3

def get_team_context(team_id, standings_df, league_cfg, matchday, title_decided=False):
    if standings_df.empty:
        return {"label":"Sin datos","emoji":"❓","css":"ctx-mid","alert":False,"dead":False}
    row = standings_df[standings_df["team_id"]==team_id]
    if row.empty:
        return {"label":"Sin datos","emoji":"❓","css":"ctx-mid","alert":False,"dead":False}
    row = row.iloc[0]
    pos = int(row["position"])

    # ── Torneo (Mundial / Euros) — contexto por posición en grupo ──
    if league_cfg.get("is_tournament", False):
        played = int(row["played"])
        pts    = int(row["points"])
        group  = row.get("group", "")
        group_label = f"Grupo {group.split('_')[-1]}" if group else "Grupo"
        if pos == 1:
            return {"label":f"Líder {group_label}","emoji":"🥇","css":"ctx-champ-won","alert":True,"dead":False}
        elif pos == 2:
            return {"label":f"Clasifica {group_label}","emoji":"✅","css":"ctx-champion","alert":True,"dead":False}
        elif pos == 3:
            # Tercer puesto — puede clasificar como mejor 3º
            return {"label":"Mejor 3º (borde)","emoji":"⚠️","css":"ctx-nearrel","alert":True,"dead":False}
        else:
            return {"label":"En eliminación","emoji":"🔴","css":"ctx-relegation","alert":True,"dead":False}

    pos       = int(row["position"])
    pts       = int(row["points"])
    played    = int(row["played"])
    total_g   = league_cfg["games"]
    remaining = max(0, total_g - played)
    cl_spots  = league_cfg["cl"]
    euro_spots= league_cfg["euro"]
    rel_spots = league_cfg["rel"]
    total_t   = league_cfg["teams"]
    safe_pos  = total_t - rel_spots
    late      = matchday >= (total_g - LATE_SEASON + 1)

    def pts_at(p):
        r = standings_df[standings_df["position"]==p]
        return int(r["points"].iloc[0]) if not r.empty else 0

    pts_rel   = pts_at(safe_pos+1)
    pts_cl    = pts_at(cl_spots)

    if pos==1 and title_decided:
        return {"label":"Campeón 🏆","emoji":"🥇","css":"ctx-champ-won","alert":False,"dead":False}
    if pos<=2 and not title_decided:
        return {"label":"Pelea título","emoji":"🏆","css":"ctx-title","alert":True,"dead":False}
    if pos<=cl_spots:
        return {"label":"Zona Champions","emoji":"⭐","css":"ctx-champion","alert":True,"dead":False}
    if pos==cl_spots+1 and pts>=pts_cl-remaining*3:
        return {"label":"Persigue Champions","emoji":"⭐","css":"ctx-champion","alert":True,"dead":False}
    if pos<=euro_spots:
        return {"label":"Zona Europa","emoji":"🌍","css":"ctx-europa","alert":True,"dead":False}
    if pos>safe_pos:
        return {"label":"Zona descenso","emoji":"🔴","css":"ctx-relegation","alert":True,"dead":False}
    if pts-pts_rel<=3:
        return {"label":"Pelea descenso","emoji":"🟠","css":"ctx-nearrel","alert":True,"dead":False}
    if late and pts>pts_rel+remaining*3 and pos>euro_spots:
        return {"label":"Sin motivación","emoji":"😴","css":"ctx-dead","alert":True,"dead":True}
    return {"label":"Zona media","emoji":"➖","css":"ctx-mid","alert":False,"dead":False}

def match_alerts(home_ctx, away_ctx, matchday, league_cfg, home_name="Local", away_name="Visitante"):
    alerts = []

    # ── Torneo: alertas de avance/eliminación ──────────────────
    if league_cfg.get("is_tournament", False):
        rel_h = home_ctx["css"] == "ctx-relegation"
        rel_a = away_ctx["css"] == "ctx-relegation"
        lim_h = home_ctx["css"] == "ctx-nearrel"
        lim_a = away_ctx["css"] == "ctx-nearrel"
        if rel_h and rel_a:
            alerts.append("🔴 Partido de eliminación directa — ambos equipos se juegan seguir en el torneo.")
        elif rel_h:
            alerts.append(f"🔴 {home_name} está en zona de eliminación — la motivación local es máxima.")
        elif rel_a:
            alerts.append(f"🔴 {away_name} está en zona de eliminación — la motivación visitante es máxima.")
        if lim_h or lim_a:
            alerts.append("⚠️ Equipo(s) en borde de clasificación — resultado con altísimo impacto en el grupo.")
        if matchday > league_cfg["games"]:
            alerts.append("⚡ Fase eliminatoria — el modelo Poisson es válido pero no hay historial de partidos a partido único.")
        return alerts

    remaining = max(0, league_cfg["games"] - matchday)
    if remaining <= 3:
        alerts.append(f"⚠️ Jornada {matchday}/{league_cfg['games']} — el modelo NO considera motivaciones ni rotaciones en estas fechas.")
    if home_ctx["dead"] and away_ctx["dead"]:
        alerts.append("😴 Ambos equipos sin motivación real — resultado muy incierto.")
    elif home_ctx["dead"]:
        alerts.append("😴 Local sin motivación extra — posibles rotaciones.")
    elif away_ctx["dead"]:
        alerts.append("😴 Visitante sin motivación extra — posibles rotaciones.")
    if home_ctx["css"]=="ctx-relegation" and away_ctx["dead"]:
        alerts.append("🔴 Descenso vs sin motivación — altísima varianza.")
    elif away_ctx["css"]=="ctx-relegation" and home_ctx["dead"]:
        alerts.append("🔴 Sin motivación vs descenso — altísima varianza.")
    return alerts

# ═══════════════════════════════════════════════════════════
# MODELO POISSON
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def build_ratings(df, decay_rate=None, intl_mode=False):
    """
    Construye ratings de ataque/defensa con shrinkage bayesiano.

    decay_rate  : tasa de decaimiento exponencial. Si None usa DECAY_RATE (ligas)
                  o INTL_DECAY (selecciones).
    intl_mode   : si True, usa comp_weight por partido (columna del DataFrame)
                  y baja el umbral mínimo de partidos a 2.
    """
    if decay_rate is None:
        decay_rate = INTL_DECAY if intl_mode else DECAY_RATE

    min_games = 2 if intl_mode else 4
    min_rows  = 5 if intl_mode else 20
    if df.empty or len(df) < min_rows:
        return {}, 1.35, 1.10

    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["days_ago"] = (now - df["date_parsed"]).dt.days.clip(lower=0)
    df["time_w"]   = np.exp(-decay_rate * df["days_ago"])

    # Peso final = decaimiento temporal × peso competitivo
    if intl_mode and "comp_weight" in df.columns:
        df["w"] = df["time_w"] * df["comp_weight"].fillna(1.0)
    else:
        df["w"] = df["time_w"]

    avg_h = float(np.average(df["home_goals"].values, weights=df["w"].values))
    avg_a = float(np.average(df["away_goals"].values, weights=df["w"].values))
    if avg_h == 0 or avg_a == 0:
        return {}, 1.35, 1.10

    mean_w = df["w"].mean()
    def wavg(vals, wts, fb=1.0):
        s = wts.sum()
        return float(np.average(vals, weights=wts)) if (len(vals) >= 1 and s > 0) else fb
    def shrink(raw, n):
        return (n * raw + SHRINKAGE_K * 1.0) / (n + SHRINKAGE_K)

    ratings = {}
    for tid in set(df["home_id"]) | set(df["away_id"]):
        hm = df[df["home_id"] == tid]
        am = df[df["away_id"] == tid]
        nh, na = len(hm), len(am)
        if nh + na < min_games:
            continue
        nh_eff = hm["w"].sum() / mean_w if nh >= 1 else 0.0
        na_eff = am["w"].sum() / mean_w if na >= 1 else 0.0
        att_h = shrink(wavg(hm["home_goals"].values, hm["w"].values) / avg_h if nh >= 2 else 1.0, nh_eff)
        att_a = shrink(wavg(am["away_goals"].values, am["w"].values) / avg_a if na >= 2 else 1.0, na_eff)
        def_h = shrink(wavg(hm["away_goals"].values, hm["w"].values) / avg_a if nh >= 2 else 1.0, nh_eff)
        def_a = shrink(wavg(am["home_goals"].values, am["w"].values) / avg_h if na >= 2 else 1.0, na_eff)
        gs_avg = (hm["home_goals"].sum() + am["away_goals"].sum()) / (nh + na)
        gc_avg = (hm["away_goals"].sum() + am["home_goals"].sum()) / (nh + na)

        # Meta: fuentes de datos usadas (para mostrar en análisis)
        src_labels = []
        if intl_mode and "comp_label" in df.columns:
            comps_used = pd.concat([hm, am])["comp_label"].dropna().unique().tolist()
            src_labels = comps_used

        ratings[tid] = {
            "att_h":   round(att_h, 3),   "att_a":   round(att_a, 3),
            "def_h":   round(def_h, 3),   "def_a":   round(def_a, 3),
            "gs_avg":  round(gs_avg, 2),  "gc_avg":  round(gc_avg, 2),
            "n":       nh + na,            "n_eff":   round(nh_eff + na_eff, 1),
            "src_labels": src_labels,
        }
    return ratings, avg_h, avg_a

def build_stat_ratings(df, home_col, away_col, decay_rate=0.008, min_games=4):
    """Ratings genéricos de ataque/defensa para corners, tiros o tiros al arco."""
    df = df.dropna(subset=[home_col, away_col, "home_id", "away_id"]).copy()
    if len(df) < 10:
        avg_h = float(df[home_col].mean()) if len(df) > 0 else 5.0
        avg_a = float(df[away_col].mean()) if len(df) > 0 else 4.5
        return {}, avg_h, avg_a

    df["date_parsed"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date_parsed"])
    now = pd.Timestamp.now(tz="UTC")
    df["days_ago"] = (now - df["date_parsed"]).dt.days.clip(lower=0)
    df["w"] = np.exp(-decay_rate * df["days_ago"])

    avg_h = float(np.average(df[home_col].values, weights=df["w"].values))
    avg_a = float(np.average(df[away_col].values, weights=df["w"].values))
    if avg_h == 0 or avg_a == 0:
        return {}, 5.0, 4.5

    K = 6
    ratings = {}
    for tid in set(df["home_id"]) | set(df["away_id"]):
        hm = df[df["home_id"] == tid]
        am = df[df["away_id"] == tid]
        n = len(hm) + len(am)
        if n < min_games:
            continue
        def wavg(vals, wts, fb=1.0):
            return float(np.average(vals, weights=wts)) if (len(vals) > 0 and wts.sum() > 0) else fb
        n_eff = min(n, 40)
        raw_att_h = wavg((hm[home_col] / avg_h).values, hm["w"].values) if len(hm) >= 2 else 1.0
        raw_att_a = wavg((am[away_col] / avg_a).values, am["w"].values) if len(am) >= 2 else 1.0
        raw_def_h = wavg((hm[away_col] / avg_a).values, hm["w"].values) if len(hm) >= 2 else 1.0
        raw_def_a = wavg((am[home_col] / avg_h).values, am["w"].values) if len(am) >= 2 else 1.0
        ratings[tid] = {
            "att_h": (n_eff * raw_att_h + K) / (n_eff + K),
            "att_a": (n_eff * raw_att_a + K) / (n_eff + K),
            "def_h": (n_eff * raw_def_h + K) / (n_eff + K),
            "def_a": (n_eff * raw_def_a + K) / (n_eff + K),
            "n": n,
        }
    return ratings, avg_h, avg_a

def stat_ou_probs(home_id, away_id, ratings, avg_h, avg_a, lines, clip=(0.5, 25.0)):
    """Calcula probabilidades Over/Under para líneas dadas (corners, tiros, etc.)."""
    if home_id not in ratings or away_id not in ratings:
        return None
    hr, ar = ratings[home_id], ratings[away_id]
    lam_h = float(np.clip(hr["att_h"] * ar["def_a"] * avg_h, clip[0], clip[1]))
    lam_a = float(np.clip(ar["att_a"] * hr["def_h"] * avg_a, clip[0], clip[1]))
    lam   = lam_h + lam_a
    max_k = max(int(lam * 4) + 5, 40)
    result = {"lam_h": round(lam_h, 1), "lam_a": round(lam_a, 1), "lam_total": round(lam, 1)}
    for line in lines:
        over_p = float(sum(poisson.pmf(k, lam) for k in range(int(line) + 1, max_k)))
        key = str(line).replace(".", "_")
        result[f"o{key}"] = round(over_p, 4)
        result[f"u{key}"] = round(1 - over_p, 4)
    return result

DC_RHO = -0.13  # Dixon-Coles: corrige sobreestimación Poisson en {0-0,1-0,0-1,1-1}

def _dc_tau(x, y, lam_h, lam_a, rho):
    """Factor de corrección Dixon-Coles para marcadores bajos."""
    if   x == 0 and y == 0: return 1 - lam_h * lam_a * rho
    elif x == 0 and y == 1: return 1 + lam_h * rho
    elif x == 1 and y == 0: return 1 + lam_a * rho
    elif x == 1 and y == 1: return 1 - rho
    else:                   return 1.0

def match_probs(home_id, away_id, ratings, avg_h, avg_a):
    if home_id not in ratings or away_id not in ratings: return None
    hr,ar = ratings[home_id], ratings[away_id]
    lam_h=float(np.clip(hr["att_h"]*ar["def_a"]*avg_h,0.40,4.5))
    lam_a=float(np.clip(ar["att_a"]*hr["def_h"]*avg_a,0.40,4.5))
    G=8
    M=np.outer([poisson.pmf(i,lam_h) for i in range(G)],[poisson.pmf(i,lam_a) for i in range(G)])
    # Corrección Dixon-Coles: ajusta probabilidades de marcadores 0-0, 1-0, 0-1, 1-1
    for i in range(2):
        for j in range(2):
            M[i][j] *= _dc_tau(i, j, lam_h, lam_a, DC_RHO)
    M /= M.sum()  # renormalizar tras la corrección
    hw=float(np.sum(np.tril(M,-1))); dr=float(np.sum(np.diag(M))); aw=float(np.sum(np.triu(M,1)))
    o15=float(sum(M[i][j] for i in range(G) for j in range(G) if i+j>1))
    o25=float(sum(M[i][j] for i in range(G) for j in range(G) if i+j>2))
    o35=float(sum(M[i][j] for i in range(G) for j in range(G) if i+j>3))
    btts=float((1-poisson.pmf(0,lam_h))*(1-poisson.pmf(0,lam_a)))
    f=lambda p: round(1/p,2) if p>0.01 else 99.0
    return {"lam_h":round(lam_h,2),"lam_a":round(lam_a,2),
            "home_win":round(hw,4),"draw":round(dr,4),"away_win":round(aw,4),
            "over15":round(o15,4),"under15":round(1-o15,4),
            "over25":round(o25,4),"under25":round(1-o25,4),
            "over35":round(o35,4),"under35":round(1-o35,4),
            "btts":round(btts,4),"no_btts":round(1-btts,4),
            "fair_1":f(hw),"fair_x":f(dr),"fair_2":f(aw),
            "fair_o15":f(o15),"fair_u15":f(1-o15),
            "fair_o25":f(o25),"fair_u25":f(1-o25),
            "fair_o35":f(o35),"fair_u35":f(1-o35),
            "fair_btts":f(btts),"fair_no_btts":f(1-btts)}

def team_form(df, team_id, n=6):
    hm=df[df["home_id"]==team_id][["date","home_goals","away_goals"]].rename(columns={"home_goals":"gf","away_goals":"gc"}); hm["v"]="H"
    am=df[df["away_id"]==team_id][["date","home_goals","away_goals"]].rename(columns={"away_goals":"gf","home_goals":"gc"}); am["v"]="A"
    out=[]
    for _,r in pd.concat([hm,am]).sort_values("date",ascending=False).head(n).iterrows():
        if r.gf>r.gc: out.append(("W","fb-w",r.gf,r.gc,r.v))
        elif r.gf==r.gc: out.append(("D","fb-d",r.gf,r.gc,r.v))
        else: out.append(("L","fb-l",r.gf,r.gc,r.v))
    return out

# ═══════════════════════════════════════════════════════════
# VALUE BET ENGINE
# ═══════════════════════════════════════════════════════════
# Selecciones nacionales que distintas fuentes (football-data.org vs. casas
# de apuestas) nombran de forma diferente — normalizamos a una forma canónica
# antes de comparar para no duplicar el mismo partido en la lista.
COUNTRY_ALIASES = {
    "czech republic": "czechia", "czechia": "czechia",
    "south korea": "korea republic", "korea republic": "korea republic", "korea": "korea republic",
    "usa": "united states", "united states": "united states", "us soccer": "united states",
    "ivory coast": "cote d'ivoire", "côte d'ivoire": "cote d'ivoire", "cote d'ivoire": "cote d'ivoire",
    "bosnia and herzegovina": "bosnia-herzegovina", "bosnia-herzegovina": "bosnia-herzegovina", "bosnia": "bosnia-herzegovina",
    "north macedonia": "macedonia", "macedonia": "macedonia",
    "iran": "ir iran", "ir iran": "ir iran",
    "cape verde": "cabo verde", "cabo verde": "cabo verde",
    "dr congo": "congo dr", "congo dr": "congo dr", "democratic republic of the congo": "congo dr",
    "united kingdom": "great britain", "great britain": "great britain",
    "england": "england", "republic of ireland": "ireland", "ireland": "ireland",
}

def _norm_country(name):
    n = str(name or "").lower().strip()
    return COUNTRY_ALIASES.get(n, n)

def _sim(a, b):
    # Guard: convertir a string si no lo son (None, int, etc.)
    a = str(a) if a is not None else ""
    b = str(b) if b is not None else ""
    a, b = _norm_country(a), _norm_country(b)
    for s in [" fc"," cf"," afc"," sc"," united"]:
        a = a.lower().replace(s, "")
        b = b.lower().replace(s, "")
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

def find_odds_match(fd_home, fd_away, fd_date_str, odds_list):
    try:
        fd_date = datetime.fromisoformat(fd_date_str.replace("Z", "+00:00")).date()
    except Exception:
        return None
    best, bs = None, 0
    for om in odds_list:
        # Saltar entradas sin nombre de equipo (datos incompletos de la API)
        if not om.get("home_team") or not om.get("away_team"):
            continue
        try:
            om_date = datetime.fromisoformat(om["commence_time"].replace("Z", "+00:00")).date()
        except Exception:
            continue
        if abs((om_date - fd_date).days) > 1:
            continue
        try:
            score = _sim(fd_home, om["home_team"]) + _sim(fd_away, om["away_team"])
        except Exception:
            continue
        if score > bs and score > 1.25:
            bs, best = score, om
    return best

def best_odds_for(om):
    best={"h2h_1":0,"h2h_x":0,"h2h_2":0,"o15":0,"u15":0,"o25":0,"u25":0,"o35":0,"u35":0,
          "btts_yes":0,"btts_no":0,
          "c85":0,"c95":0,"c105":0,"c115":0,"uc85":0,"uc95":0,"uc105":0,"uc115":0}
    if not om: return best
    ht,at=om.get("home_team",""),om.get("away_team","")
    for bk in om.get("bookmakers",[]):
        for mkt in bk.get("markets",[]):
            if mkt["key"]=="h2h":
                for oc in mkt.get("outcomes",[]):
                    p,n=float(oc.get("price",0)),oc.get("name","")
                    if   _sim(n,ht)>0.7: best["h2h_1"]=max(best["h2h_1"],p)
                    elif _sim(n,at)>0.7: best["h2h_2"]=max(best["h2h_2"],p)
                    elif "draw" in n.lower(): best["h2h_x"]=max(best["h2h_x"],p)
            elif mkt["key"]=="totals":
                for oc in mkt.get("outcomes",[]):
                    pt,pr=float(oc.get("point",0)),float(oc.get("price",0))
                    nm=oc.get("name","").lower()
                    if   abs(pt-1.5)<0.01:
                        if nm=="over":  best["o15"]=max(best["o15"],pr)
                        else:           best["u15"]=max(best["u15"],pr)
                    elif abs(pt-2.5)<0.01:
                        if nm=="over":  best["o25"]=max(best["o25"],pr)
                        else:           best["u25"]=max(best["u25"],pr)
                    elif abs(pt-3.5)<0.01:
                        if nm=="over":  best["o35"]=max(best["o35"],pr)
                        else:           best["u35"]=max(best["u35"],pr)
            elif mkt["key"] in ("btts","both_teams_to_score"):
                for oc in mkt.get("outcomes",[]):
                    p,n=float(oc.get("price",0)),oc.get("name","").lower()
                    if "yes" in n: best["btts_yes"]=max(best["btts_yes"],p)
                    elif "no" in n: best["btts_no"]=max(best["btts_no"],p)
            elif mkt["key"] in ("bookie_corners","corners","total_corners"):
                for oc in mkt.get("outcomes",[]):
                    pt = float(oc.get("point", 0))
                    pr = float(oc.get("price", 0))
                    nm = oc.get("name","").lower()
                    line_map = {8.5:("c85","uc85"), 9.5:("c95","uc95"),
                                10.5:("c105","uc105"), 11.5:("c115","uc115")}
                    if pt in line_map:
                        ok, uk = line_map[pt]
                        if nm == "over":  best[ok]  = max(best[ok],  pr)
                        elif nm == "under": best[uk] = max(best[uk], pr)
    return best

def kelly_criterion(model_p, bk_odds, fraction=0.25):
    b = bk_odds - 1
    q = 1 - model_p
    k = (model_p * b - q) / b
    return round(max(0.0, k) * fraction * 100, 1)

def conf_info(edge):
    # Mayor edge = más valor detectado = mayor confianza
    if   edge>=0.12: return "Alta",  "high",  "🟢","ev-high","conf-high"
    elif edge>=0.07: return "Media", "medium","🟡","ev-medium","conf-medium"
    else:            return "Baja",  "low",   "🟠","ev-low","conf-low"

def detect_value_bets(probs, bk, home_name, away_name, ev_threshold, corner_probs=None):
    if not probs: return []
    checks=[
        ("1 Local",       probs["home_win"],  bk["h2h_1"]),
        ("X Empate",      probs["draw"],       bk["h2h_x"]),
        ("2 Visitante",   probs["away_win"],   bk["h2h_2"]),
        ("Over 1.5",      probs["over15"],     bk["o15"]),
        ("Under 1.5",     probs["under15"],    bk["u15"]),
        ("Over 2.5",      probs["over25"],     bk["o25"]),
        ("Under 2.5",     probs["under25"],    bk["u25"]),
        ("Over 3.5",      probs["over35"],     bk["o35"]),
        ("Under 3.5",     probs["under35"],    bk["u35"]),
        ("BTTS Sí",       probs["btts"],       bk["btts_yes"]),
        ("BTTS No",       probs["no_btts"],    bk["btts_no"]),
    ]
    if corner_probs:
        checks += [
            ("Córners +8.5",  corner_probs.get("o8_5", 0),  bk["c85"]),
            ("Córners -8.5",  corner_probs.get("u8_5", 0),  bk["uc85"]),
            ("Córners +9.5",  corner_probs.get("o9_5", 0),  bk["c95"]),
            ("Córners -9.5",  corner_probs.get("u9_5", 0),  bk["uc95"]),
            ("Córners +10.5", corner_probs.get("o10_5", 0), bk["c105"]),
            ("Córners -10.5", corner_probs.get("u10_5", 0), bk["uc105"]),
            ("Córners +11.5", corner_probs.get("o11_5", 0), bk["c115"]),
            ("Córners -11.5", corner_probs.get("u11_5", 0), bk["uc115"]),
        ]
    found=[]
    for label,model_p,bk_odd in checks:
        if bk_odd<MIN_ODDS or bk_odd>MAX_ODDS: continue
        ev=model_p*bk_odd-1
        if ev<ev_threshold: continue
        implied=1/bk_odd; edge=model_p-implied
        if edge>MAX_EDGE: continue
        cl,ck,ci,ev_css,conf_css=conf_info(edge)
        found.append({"label":label,"model_p":round(model_p,4),"implied":round(implied,4),
                      "edge":round(edge,4),"bk_odds":bk_odd,"ev":round(ev,4),
                      "conf_label":cl,"conf_key":ck,"conf_icon":ci,
                      "ev_css":ev_css,"conf_css":conf_css,
                      "kelly":kelly_criterion(model_p, bk_odd),
                      "home":home_name,"away":away_name})
    return found

def suggest_top_market(probs, bk):
    """
    Cuando el modelo no detecta ningún 'value bet' (EV insuficiente), igual
    queremos mostrarle al usuario cuál es el mercado que el modelo considera
    más probable —siempre que la casa de apuestas ofrezca cuota para él—
    para que cada partido tenga al menos una sugerencia de referencia.
    Esto NO es un value bet (no implica edge positivo): se etiqueta aparte.
    """
    if not probs:
        return None
    candidates = [
        ("1 Local",     probs["home_win"], bk.get("h2h_1", 0), probs.get("fair_1")),
        ("X Empate",    probs["draw"],     bk.get("h2h_x", 0), probs.get("fair_x")),
        ("2 Visitante", probs["away_win"], bk.get("h2h_2", 0), probs.get("fair_2")),
        ("Over 2.5",    probs["over25"],   bk.get("o25", 0),   probs.get("fair_o25")),
        ("Under 2.5",   probs["under25"],  bk.get("u25", 0),   probs.get("fair_u25")),
    ]
    candidates = [c for c in candidates if c[2] and c[2] > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[1], reverse=True)
    label, model_p, odd, fair = candidates[0]
    implied = 1 / odd if odd else 0
    return {
        "label": label, "model_p": round(model_p, 4), "bk_odds": odd,
        "fair": fair, "implied": round(implied, 4),
        "edge": round(model_p - implied, 4),
    }

# ═══════════════════════════════════════════════════════════
# TRACKER — PERSISTENCIA Y ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════
def load_history():
    try:
        res = get_supabase().table("picks").select("*").order("created_at", desc=False).execute()
        return res.data or []
    except Exception:
        return []

def add_pick_to_history(home, away, league, market, odds, model_p, ev, stake, date_match, closing_odds=None):
    co = round(float(closing_odds), 2) if closing_odds and float(closing_odds) > 1.0 else None
    clv = round((float(odds) / co - 1) * 100, 1) if co else None
    pick = {
        "id":           str(uuid.uuid4())[:8],
        "created_at":   datetime.utcnow().isoformat(),
        "date_match":   date_match,
        "home": home,   "away": away,
        "league": league, "market": market,
        "odds":         round(float(odds), 2),
        "closing_odds": co,
        "clv":          clv,
        "model_p":      round(float(model_p)*100, 1),
        "ev":           round(float(ev)*100, 1),
        "stake":        round(float(stake), 2) if stake else 0.0,
        "result":       "pending",
        "resolved_at":  None, "pnl": None,
    }
    get_supabase().table("picks").insert(pick).execute()
    return pick["id"]

def resolve_pick(pick_id, result, stake=None, closing_odds=None):
    history = load_history()
    p = next((x for x in history if x["id"] == pick_id), None)
    if not p:
        return
    stake_val = round(float(stake), 2) if stake is not None else p["stake"]
    pnl = round(stake_val * (p["odds"] - 1), 2) if result == "hit" else -stake_val
    update = {
        "result":      result,
        "resolved_at": datetime.utcnow().isoformat(),
        "stake":       stake_val,
        "pnl":         pnl,
    }
    if closing_odds and float(closing_odds) > 1.0:
        co = round(float(closing_odds), 2)
        update["closing_odds"] = co
        update["clv"] = round((p["odds"] / co - 1) * 100, 1)
    get_supabase().table("picks").update(update).eq("id", pick_id).execute()

def delete_pick(pick_id):
    get_supabase().table("picks").delete().eq("id", pick_id).execute()

def get_streak(history):
    resolved = sorted(
        [p for p in history if p["result"] in ("hit","miss")],
        key=lambda x: x.get("resolved_at") or "", reverse=True
    )
    if not resolved: return 0, None
    kind = resolved[0]["result"]
    count = sum(1 for p in resolved if p["result"] == kind)
    # Only consecutive from latest
    streak = 0
    for p in resolved:
        if p["result"] == kind: streak += 1
        else: break
    return streak, kind

def get_stats(history):
    resolved = [p for p in history if p["result"] in ("hit","miss")]
    if not resolved:
        return {"total":0,"hits":0,"misses":0,"win_rate":0.0,"roi":0.0,"pnl":0.0,"staked":0.0}
    hits    = sum(1 for p in resolved if p["result"]=="hit")
    pnl     = sum(p.get("pnl") or 0 for p in resolved)
    staked  = sum(p.get("stake") or 0 for p in resolved)
    clv_picks = [p["clv"] for p in resolved if p.get("clv") is not None]
    avg_clv = round(sum(clv_picks) / len(clv_picks), 1) if clv_picks else None
    return {
        "total":    len(resolved),
        "hits":     hits,
        "misses":   len(resolved)-hits,
        "win_rate": round(hits/len(resolved)*100, 1),
        "roi":      round(pnl/staked*100, 1) if staked > 0 else 0.0,
        "pnl":      round(pnl, 2),
        "staked":   round(staked, 2),
        "avg_clv":  avg_clv,
        "n_clv":    len(clv_picks),
    }

def streak_html(streak_count, streak_type):
    if streak_count == 0 or streak_type is None:
        return (
            '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:18px;text-align:center">'
            '<div style="font-size:.65rem;letter-spacing:1px;color:#94a3b8;font-family:\'IBM Plex Mono\',monospace;margin-bottom:6px">RACHA ACTIVA</div>'
            '<div style="font-size:1.8rem;color:#94a3b8">—</div>'
            '<div style="font-size:.72rem;color:#94a3b8;margin-top:4px">Sin historial resuelto</div>'
            '</div>'
        )
    is_hit  = streak_type == "hit"
    icon    = "🔥" if is_hit else "🧊"
    label   = "HITS CONSECUTIVOS" if is_hit else "MISSES CONSECUTIVOS"
    color   = BRAND_GREEN if is_hit else "#ef4444"
    bg      = "rgba(0,168,107,0.07)" if is_hit else "rgba(239,68,68,0.07)"
    border  = "rgba(0,168,107,0.25)" if is_hit else "rgba(239,68,68,0.25)"
    icons   = "".join(f'<span style="font-size:1.3rem;opacity:{max(0.25,1-i*0.14)}">{icon}</span>' for i in range(min(streak_count, 6)))
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:18px;text-align:center">'
        f'<div style="font-size:.63rem;letter-spacing:1.2px;color:{color};font-family:\'IBM Plex Mono\',monospace;margin-bottom:6px">{label}</div>'
        f'<div style="font-size:3rem;font-weight:700;color:{BRAND_DARK};font-family:\'IBM Plex Mono\',monospace;line-height:1">{streak_count}</div>'
        f'<div style="margin-top:8px">{icons}</div>'
        f'</div>'
    )

def calendar_html_grid(history, year, month):
    picks_by_date = {}
    for p in history:
        raw = p.get("resolved_at") or p.get("created_at") or ""
        date_str = raw[:10]
        if date_str: picks_by_date.setdefault(date_str, []).append(p)

    month_name = cal_module.month_name[month].upper()
    matrix     = cal_module.monthcalendar(year, month)
    today_str  = datetime.utcnow().strftime("%Y-%m-%d")

    # Day headers
    hdr = "".join(
        f'<div style="text-align:center;font-size:.62rem;color:#94a3b8;font-family:\'IBM Plex Mono\',monospace;padding-bottom:4px">{d}</div>'
        for d in ["L","M","X","J","V","S","D"]
    )

    cells = ""
    for week in matrix:
        for day in week:
            if day == 0:
                cells += '<div></div>'; continue
            ds = f"{year}-{month:02d}-{day:02d}"
            dp = picks_by_date.get(ds, [])
            is_today = ds == today_str

            if not dp:
                bg, border_c = "#f8fafc", "#e2e8f0"
                dot = ""
            else:
                hits    = sum(1 for p in dp if p["result"]=="hit")
                misses  = sum(1 for p in dp if p["result"]=="miss")
                pending = sum(1 for p in dp if p["result"]=="pending")
                n       = len(dp)
                if hits and not misses:
                    bg, border_c = "rgba(0,168,107,0.13)", "rgba(0,168,107,0.35)"
                elif misses and not hits:
                    bg, border_c = "rgba(239,68,68,0.11)", "rgba(239,68,68,0.30)"
                elif hits and misses:
                    bg, border_c = "rgba(217,119,6,0.11)", "rgba(217,119,6,0.30)"
                else:
                    bg, border_c = "rgba(26,111,168,0.09)", "rgba(26,111,168,0.25)"
                dot = f'<div style="font-size:.58rem;color:#64748b;font-family:\'IBM Plex Mono\',monospace">{n}p</div>'

            today_ring = "box-shadow:0 0 0 2px #00A86B;" if is_today else ""
            cells += (
                f'<div style="background:{bg};border:1px solid {border_c};border-radius:6px;'
                f'padding:5px 2px;text-align:center;min-height:42px;{today_ring}">'
                f'<div style="font-size:.72rem;font-weight:600;color:{BRAND_DARK};font-family:\'IBM Plex Mono\',monospace">{day}</div>'
                f'{dot}</div>'
            )

    legend = (
        '<div style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap">'
        + "".join(
            f'<div style="display:flex;align-items:center;gap:4px;font-size:.65rem;color:#64748b">'
            f'<div style="width:9px;height:9px;border-radius:2px;background:{bg};border:1px solid {bc}"></div>{lbl}</div>'
            for lbl,bg,bc in [
                ("Hit","rgba(0,168,107,0.15)","rgba(0,168,107,0.35)"),
                ("Miss","rgba(239,68,68,0.12)","rgba(239,68,68,0.30)"),
                ("Mixto","rgba(217,119,6,0.12)","rgba(217,119,6,0.30)"),
                ("Pendiente","rgba(26,111,168,0.10)","rgba(26,111,168,0.25)"),
            ]
        )
        + '</div>'
    )

    return (
        f'<div style="background:white;border-radius:14px;padding:18px 18px 14px;border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(0,0,0,0.05)">'
        f'<div style="font-size:.72rem;font-weight:700;color:{BRAND_DARK};font-family:\'Space Grotesk\',sans-serif;letter-spacing:.5px;margin-bottom:12px">{month_name} {year}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:4px;margin-bottom:4px">{hdr}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:4px">{cells}</div>'
        f'{legend}'
        f'</div>'
    )

# ═══════════════════════════════════════════════════════════
# CALENDARIO DE PARTIDOS — selector visual de fechas
# ═══════════════════════════════════════════════════════════
def match_dates_calendar_html(upcoming, selected_date_str=None):
    """
    Genera un calendario HTML compacto con los meses que tienen partidos.
    Las fechas con partidos llevan un punto verde y fondo suave.
    La fecha seleccionada queda en verde sólido.
    """
    if not upcoming:
        return ""

    match_counts: dict = {}
    for m in upcoming:
        d = m["date"][:10]
        match_counts[d] = match_counts.get(d, 0) + 1

    if not match_counts:
        return ""

    dates_sorted = sorted(match_counts)
    today        = datetime.utcnow().date()

    # Meses a mostrar (desde primer partido hasta último)
    first_dt = datetime.strptime(dates_sorted[0],  "%Y-%m-%d").date().replace(day=1)
    last_dt  = datetime.strptime(dates_sorted[-1], "%Y-%m-%d").date().replace(day=1)
    months   = []
    cur = first_dt
    while cur <= last_dt:
        months.append((cur.year, cur.month))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    hdr_row = "".join(
        f'<div style="text-align:center;font-size:.58rem;color:#94a3b8;'
        f'font-family:IBM Plex Mono,monospace;padding-bottom:3px">{d}</div>'
        for d in ["L","M","X","J","V","S","D"]
    )

    month_blocks = []
    for year, month in months:
        month_label = cal_module.month_name[month].upper()
        matrix      = cal_module.monthcalendar(year, month)
        cells       = ""
        for week in matrix:
            for day in week:
                if day == 0:
                    cells += '<div></div>'
                    continue
                ds        = f"{year}-{month:02d}-{day:02d}"
                n         = match_counts.get(ds, 0)
                is_today  = datetime.strptime(ds, "%Y-%m-%d").date() == today
                is_sel    = ds == selected_date_str

                if is_sel:
                    bg = "#00A86B"; txt_c = "white"; border = ""
                elif n:
                    bg = "rgba(0,168,107,0.11)"; txt_c = "#0A0D12"
                    border = "border:1.5px solid rgba(0,168,107,0.40);"
                elif is_today:
                    bg = "rgba(26,111,168,0.09)"; txt_c = "#0A0D12"
                    border = "border:1.5px solid rgba(26,111,168,0.35);"
                else:
                    bg = "transparent"; txt_c = "#94a3b8"; border = "border:1px solid transparent;"

                dot = (
                    '<div style="width:4px;height:4px;border-radius:50%;'
                    'background:#00A86B;margin:1px auto 0"></div>'
                ) if n and not is_sel else ""
                fw  = "700" if (n or is_sel) else "400"

                cells += (
                    f'<div style="background:{bg};{border}border-radius:5px;'
                    f'padding:3px 1px;text-align:center;min-height:30px">'
                    f'<div style="font-size:.68rem;font-weight:{fw};color:{txt_c};'
                    f'font-family:IBM Plex Mono,monospace;line-height:1.2">{day}</div>'
                    f'{dot}</div>'
                )

        month_blocks.append(
            f'<div style="margin-bottom:14px">'
            f'<div style="font-size:.63rem;font-weight:700;color:#0A0D12;'
            f'font-family:Space Grotesk,sans-serif;letter-spacing:.5px;margin-bottom:6px">'
            f'{month_label} {year}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:2px">{hdr_row}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px">{cells}</div>'
            f'</div>'
        )

    return (
        f'<div style="background:white;border-radius:14px;padding:16px 14px 10px;'
        f'border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.04)">'
        + "".join(month_blocks)
        + f'<div style="font-size:.60rem;color:#94a3b8;margin-top:2px">'
          f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
          f'background:#00A86B;margin-right:4px"></span>días con partidos</div>'
        + '</div>'
    )

def _sidebar_cal_html(year, month, match_counts: dict, selected_date_str=None):
    """
    Grid mensual compacto para la barra lateral.
    Días con partidos: fondo verde suave + punto.
    Día seleccionado: verde sólido.
    Hoy: aro azul.
    """
    today = datetime.utcnow().date()
    matrix = cal_module.monthcalendar(year, month)

    hdr = "".join(
        f'<div style="text-align:center;font-size:.58rem;color:#94a3b8;'
        f'font-family:IBM Plex Mono,monospace;padding-bottom:4px">{d}</div>'
        for d in ["L","M","X","J","V","S","D"]
    )
    cells = ""
    for week in matrix:
        for day in week:
            if day == 0:
                cells += '<div></div>'
                continue
            ds = f"{year}-{month:02d}-{day:02d}"
            n  = match_counts.get(ds, 0)
            is_today = datetime.strptime(ds, "%Y-%m-%d").date() == today
            is_sel   = ds == selected_date_str

            if is_sel:
                bg = "#00A86B"; txt_c = "white"; extra = ""
            elif n:
                bg = "rgba(0,168,107,0.12)"; txt_c = "#0A0D12"
                extra = "border:1.5px solid rgba(0,168,107,0.40);"
            elif is_today:
                bg = "rgba(26,111,168,0.09)"; txt_c = "#1a6fa8"
                extra = "border:1.5px solid rgba(26,111,168,0.35);"
            else:
                bg = "transparent"; txt_c = "#94a3b8"; extra = "border:1px solid transparent;"

            dot = (
                '<div style="width:4px;height:4px;border-radius:50%;background:#00A86B;'
                'margin:1px auto 0"></div>'
            ) if n and not is_sel else ""
            fw = "700" if (n or is_sel) else "400"

            cells += (
                f'<div style="background:{bg};{extra}border-radius:5px;padding:3px 1px;'
                f'text-align:center;min-height:30px">'
                f'<div style="font-size:.68rem;font-weight:{fw};color:{txt_c};'
                f'font-family:IBM Plex Mono,monospace;line-height:1.3">{day}</div>'
                f'{dot}</div>'
            )

    return (
        f'<div style="padding:4px 0 8px">'
        f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:2px">{hdr}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px">{cells}</div>'
        f'<div style="display:flex;gap:10px;margin-top:8px;flex-wrap:wrap">'
        f'<div style="display:flex;align-items:center;gap:4px;font-size:.60rem;color:#64748b">'
        f'<div style="width:8px;height:8px;border-radius:2px;background:rgba(0,168,107,0.15);'
        f'border:1px solid rgba(0,168,107,0.40)"></div>Partidos</div>'
        f'<div style="display:flex;align-items:center;gap:4px;font-size:.60rem;color:#64748b">'
        f'<div style="width:8px;height:8px;border-radius:2px;background:#00A86B"></div>Seleccionado</div>'
        f'</div></div>'
    )

# ═══════════════════════════════════════════════════════════
# ANÁLISIS DE PICK ("¿Por qué este pick?")
# ═══════════════════════════════════════════════════════════
def generate_analysis(vb, ratings):
    """Genera explicación textual en Markdown del razonamiento del modelo para un pick."""
    label    = vb["label"]
    p        = vb.get("probs")
    hr       = ratings.get(vb.get("home_id"), {})
    ar       = ratings.get(vb.get("away_id"), {})
    home     = vb["home"]
    away     = vb["away"]
    model_p  = vb["model_p"] * 100
    impl_p   = vb["implied"] * 100
    edge_pp  = vb["edge"] * 100
    bk_odd   = vb["bk_odds"]
    fair     = round(1 / vb["model_p"], 2) if vb["model_p"] > 0 else 99

    att_h = hr.get("att_h", 1.0)
    att_a = ar.get("att_a", 1.0)
    def_h = hr.get("def_h", 1.0)
    def_a = ar.get("def_a", 1.0)
    gs_h  = hr.get("gs_avg", 1.35)
    gs_a  = ar.get("gs_avg", 1.10)
    gc_h  = hr.get("gc_avg", 1.10)
    gc_a  = ar.get("gc_avg", 1.35)
    n_h   = hr.get("n_eff", 0)
    n_a   = ar.get("n_eff", 0)

    lam_h = p["lam_h"] if p else 1.35
    lam_a = p["lam_a"] if p else 1.10
    xg_total = round(lam_h + lam_a, 2)

    def rating_desc(val, low_is_good=False):
        if low_is_good:
            if val < 0.88: return "muy sólido ✅"
            elif val < 0.97: return "sólido ✅"
            elif val <= 1.05: return "promedio ➡️"
            elif val <= 1.15: return "poroso ⚠️"
            else: return "muy poroso 🔴"
        else:
            if val > 1.20: return "muy superior 🔥"
            elif val > 1.07: return "superior 📈"
            elif val >= 0.95: return "promedio ➡️"
            elif val >= 0.85: return "inferior 📉"
            else: return "muy inferior 🔴"

    # Fuentes de datos (para selecciones)
    h_srcs = ", ".join(hr.get("src_labels", [])) or "datos de liga"
    a_srcs = ", ".join(ar.get("src_labels", [])) or "datos de liga"
    is_intl = bool(hr.get("src_labels"))

    def sample_note(n_eff, srcs=""):
        src_txt = f" · Fuentes: *{srcs}*" if srcs else ""
        if n_eff < 5:
            return f"⚠️ muestra muy pequeña ({n_eff:.1f} partidos efectivos) — rating fuertemente ajustado al promedio.{src_txt}"
        elif n_eff < 12:
            return f"muestra moderada ({n_eff:.1f} partidos efectivos).{src_txt}"
        else:
            return f"muestra sólida ({n_eff:.1f} partidos efectivos).{src_txt}"

    lines = []

    # ── Market-specific reasoning ──────────────────────────
    ref = "torneo" if is_intl else "liga"

    if label == "1 Local":
        lines.append(f"#### 🏠 Análisis: Victoria de {home}")
        lines.append(
            f"**Ataque local de {home}:** `{att_h:.3f}x` la media de {ref} — {rating_desc(att_h)}. "
            f"Promedia **{gs_h:.2f} goles/partido** en casa."
        )
        lines.append(
            f"**Defensa visitante de {away}:** `{def_a:.3f}x` la media — {rating_desc(def_a, low_is_good=True)}. "
            f"Concede **{gc_a:.2f} goles/partido** como visitante."
        )
        lines.append(
            f"**xG esperados:** {home} generará ~**{lam_h}** goles vs ~**{lam_a}** de {away}. "
            f"El modelo favorece al local con ventaja de **{round(lam_h-lam_a, 2):+.2f} xG**."
        )

    elif label == "2 Visitante":
        lines.append(f"#### ✈️ Análisis: Victoria de {away}")
        lines.append(
            f"**Ataque visitante de {away}:** `{att_a:.3f}x` la media de liga — {rating_desc(att_a)}. "
            f"Promedia **{gs_a:.2f} goles/partido** fuera de casa."
        )
        lines.append(
            f"**Defensa local de {home}:** `{def_h:.3f}x` la media — {rating_desc(def_h, low_is_good=True)}. "
            f"Concede **{gc_h:.2f} goles/partido** en casa."
        )
        lines.append(
            f"**xG esperados:** {away} generará ~**{lam_a}** goles fuera vs ~**{lam_h}** del local. "
            f"El modelo da ventaja al visitante de **{round(lam_a-lam_h, 2):+.2f} xG**."
        )

    elif label == "X Empate":
        balance = abs(lam_h - lam_a)
        bal_desc = "muy equilibrado ⚖️" if balance < 0.25 else ("equilibrado" if balance < 0.50 else "con ligero desequilibrio")
        lines.append(f"#### ⚖️ Análisis: Empate")
        lines.append(
            f"**Equilibrio de fuerzas:** xG {home} **{lam_h}** vs {away} **{lam_a}** — {bal_desc} "
            f"(diferencia de {balance:.2f} goles esperados)."
        )
        lines.append(
            f"**Defensa local {home}:** `{def_h:.3f}x` — {rating_desc(def_h, low_is_good=True)}. "
            f"**Defensa visit. {away}:** `{def_a:.3f}x` — {rating_desc(def_a, low_is_good=True)}."
        )
        lines.append(
            f"**xG total:** {xg_total} goles esperados — "
            f"{'partidos de bajo marcador favorecen el empate en distribución Poisson.' if xg_total < 2.4 else 'la similitud de las fuerzas aumenta la probabilidad de empate.'}"
        )

    elif label == "Over 2.5":
        lines.append(f"#### ⚽ Análisis: Over 2.5 goles")
        lines.append(
            f"**xG combinados:** {lam_h} + {lam_a} = **{xg_total}** goles esperados — "
            f"{'muy por encima del umbral 2.5 ✅' if xg_total > 3.2 else 'por encima del umbral 2.5 ✅' if xg_total > 2.5 else 'cerca del umbral ⚠️'}."
        )
        lines.append(
            f"**Potencia ofensiva {home}:** `{att_h:.3f}x` en casa — {rating_desc(att_h)}, {gs_h:.2f} goles/pj."
        )
        lines.append(
            f"**Potencia ofensiva {away}:** `{att_a:.3f}x` fuera — {rating_desc(att_a)}, {gs_a:.2f} goles/pj."
        )

    elif label == "Under 2.5":
        lines.append(f"#### 🛡️ Análisis: Under 2.5 goles")
        lines.append(
            f"**xG combinados:** {lam_h} + {lam_a} = **{xg_total}** goles esperados — "
            f"{'claramente bajo el umbral 2.5 ✅' if xg_total < 2.0 else 'por debajo del umbral 2.5 ✅' if xg_total < 2.5 else 'cerca del umbral ⚠️ — la distribución Poisson da valor al Under.'}."
        )
        lines.append(
            f"**Defensa local {home}:** `{def_h:.3f}x` — {rating_desc(def_h, low_is_good=True)}, concede {gc_h:.2f} goles/pj."
        )
        lines.append(
            f"**Defensa visit. {away}:** `{def_a:.3f}x` — {rating_desc(def_a, low_is_good=True)}, concede {gc_a:.2f} goles/pj."
        )

    # ── Veredicto estadístico ──────────────────────────────
    lines.append(
        f"**📊 Veredicto del modelo:** Probabilidad estimada **{model_p:.1f}%** "
        f"(cuota justa `{fair}`) vs cuota de mercado `{bk_odd}` (implícita {impl_p:.1f}%). "
        f"El mercado subestima esta probabilidad en **+{edge_pp:.1f} pp**."
    )

    # ── Calidad de datos ──────────────────────────────────
    lines.append(
        f"**🔬 Calidad de datos:** {home} — {sample_note(n_h, h_srcs)}  "
        f"{away} — {sample_note(n_a, a_srcs)}"
    )

    # ── Contexto motivacional ─────────────────────────────
    home_ctx = vb.get("home_ctx", {})
    away_ctx = vb.get("away_ctx", {})
    if home_ctx.get("dead") or away_ctx.get("dead"):
        who = "Local" if home_ctx.get("dead") else "Visitante"
        lines.append(
            f"**⚠️ Riesgo motivacional:** {who} en zona sin motivación — "
            f"posibles rotaciones que el modelo estadístico no puede anticipar. Considera reducir stake."
        )

    # ── Probabilidades adicionales del partido ────────────
    if p:
        o25_pct  = p["over25"] * 100
        btts_pct = p["btts"]   * 100
        lines.append(
            f"**📐 Contexto del partido:** Over 2.5 → {o25_pct:.0f}% · "
            f"BTTS → {btts_pct:.0f}% · xG local {lam_h} · xG visit. {lam_a}"
        )

    # ── CLV ───────────────────────────────────────────────
    lines.append("---")
    lines.append(
        f"*💡 **CLV Tracking:** Anota esta cuota (`{bk_odd}`). Si la cuota de cierre el día del partido "
        f"es inferior, confirma que el modelo identificó valor real. El CLV a largo plazo es el mejor "
        f"indicador de la calidad del modelo.*"
    )

    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════
def render_form(form_list):
    html=""
    for res,css,gf,gc,v in form_list:
        html+=f'<span class="form-badge {css}" title="{gf:.0f}-{gc:.0f} ({v})">{res}</span>'
    return html

def render_form_mini(form_list, n=3):
    """Mini form badges for inside cards (last n results)."""
    html=""
    for res,css,gf,gc,v in form_list[:n]:
        html+=f'<span class="form-badge {css}" title="{gf:.0f}-{gc:.0f} ({v})" style="width:20px;height:20px;line-height:20px;font-size:10px">{res}</span>'
    return html

def ctx_badge_html(ctx):
    return f'<span class="ctx-badge {ctx["css"]}">{ctx["emoji"]} {ctx["label"]}</span>'

def prob_bar_html(p, home_name, away_name):
    h=p["home_win"]*100; d=p["draw"]*100; a=p["away_win"]*100
    o=p["over25"]*100; u=p["under25"]*100; bt=p["btts"]*100
    return f"""
    <div class="prob-bar-wrap">
      <div class="prob-bar-label">Probabilidades 1 · X · 2</div>
      <div class="prob-bar-1x2">
        <div class="pb-home" style="width:{h:.1f}%">{h:.0f}%</div>
        <div class="pb-draw" style="width:{d:.1f}%">{d:.0f}%</div>
        <div class="pb-away" style="width:{a:.1f}%">{a:.0f}%</div>
      </div>
      <div class="prob-pills">
        <span class="prob-pill"><span class="prob-pill-dot pp-orange"></span>Over 2.5 <span class="prob-pill-val">{o:.0f}%</span></span>
        <span class="prob-pill"><span class="prob-pill-dot pp-purple"></span>Under 2.5 <span class="prob-pill-val">{u:.0f}%</span></span>
        <span class="prob-pill"><span class="prob-pill-dot pp-blue"></span>BTTS <span class="prob-pill-val">{bt:.0f}%</span></span>
        <span class="prob-pill"><span class="prob-pill-dot pp-green"></span>xG Local <span class="prob-pill-val">{p['lam_h']}</span></span>
        <span class="prob-pill"><span class="prob-pill-dot pp-green"></span>xG Visit. <span class="prob-pill-val">{p['lam_a']}</span></span>
      </div>
    </div>"""

def _corner_shot_detail_html(vb):
    parts = []
    cp = vb.get("corner_probs")
    sp = vb.get("shot_probs")
    sotp = vb.get("sot_probs")
    if cp:
        parts.append(
            f'<div class="vb-detail-item"><span class="vb-detail-label">🚩 Córners pred.</span>'
            f'<span class="vb-detail-val blue">{cp["lam_total"]} '
            f'<span style="font-size:.68rem;color:#94a3b8">({cp["lam_h"]} / {cp["lam_a"]})</span></span></div>'
        )
    if sp:
        parts.append(
            f'<div class="vb-detail-item"><span class="vb-detail-label">👟 Tiros pred.</span>'
            f'<span class="vb-detail-val blue">{sp["lam_total"]} '
            f'<span style="font-size:.68rem;color:#94a3b8">({sp["lam_h"]} / {sp["lam_a"]})</span></span></div>'
        )
    if sotp:
        parts.append(
            f'<div class="vb-detail-item"><span class="vb-detail-label">🎯 Al arco pred.</span>'
            f'<span class="vb-detail-val blue">{sotp["lam_total"]} '
            f'<span style="font-size:.68rem;color:#94a3b8">({sotp["lam_h"]} / {sotp["lam_a"]})</span></span></div>'
        )
    return "".join(parts)

def vb_card_html(vb, idx=0):
    date_str = fmt_match_dt(vb["date"])
    hctx, actx = vb["home_ctx"], vb["away_ctx"]
    ev_pct   = vb["ev"]*100
    edge_pp  = vb["edge"]*100
    delay    = idx * 0.07  # staggered animation

    alert_html=""
    for a in vb.get("ctx_alerts",[]):
        alert_html+=f'<div class="ctx-alert">{a}</div>'

    # Compact 1X2 probability bar — single line to avoid Streamlit markdown code-block interpretation
    prob_html = ""
    p = vb.get("probs")
    if p:
        h_pct = p["home_win"]*100
        d_pct = p["draw"]*100
        a_pct = p["away_win"]*100
        lam_h  = p["lam_h"]
        lam_a  = p["lam_a"]
        o25    = f"{p['over25']*100:.0f}%"
        btts   = f"{p['btts']*100:.0f}%"
        prob_html = (
            f'<div style="margin:12px 0 6px">'
            f'<div style="font-size:.66rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px">'
            f'xG: <b>{lam_h}</b> – <b>{lam_a}</b> &nbsp;|&nbsp; Over 2.5: <b>{o25}</b> &nbsp;|&nbsp; BTTS: <b>{btts}</b>'
            f'</div>'
            f'<div class="card-prob-bar">'
            f'<div class="cpb-home" style="width:{h_pct:.1f}%">{h_pct:.0f}%</div>'
            f'<div class="cpb-draw" style="width:{d_pct:.1f}%">{d_pct:.0f}%</div>'
            f'<div class="cpb-away" style="width:{a_pct:.1f}%">{a_pct:.0f}%</div>'
            f'</div></div>'
        )

    # Mini form row — single line
    form_h = vb.get("home_form", [])
    form_a = vb.get("away_form", [])
    form_html = ""
    if form_h or form_a:
        fh_badges = render_form_mini(form_h)
        fa_badges = render_form_mini(form_a)
        form_html = (
            f'<div style="display:flex;gap:18px;margin-top:8px;align-items:center;flex-wrap:wrap">'
            f'<div style="display:flex;align-items:center;gap:5px">'
            f'<span style="font-size:.68rem;color:#94a3b8;white-space:nowrap">Local</span>{fh_badges}'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:5px">'
            f'<span style="font-size:.68rem;color:#94a3b8;white-space:nowrap">Visit.</span>{fa_badges}'
            f'</div></div>'
        )

    xg_h = p["lam_h"] if p else "—"
    xg_a = p["lam_a"] if p else "—"
    fair  = round(1/vb["model_p"], 2)

    d_row = (
        f'<div class="vb-details">'
        f'<div class="vb-detail-item"><span class="vb-detail-label">Cuota disponible</span><span class="vb-detail-val green">📌 {vb["bk_odds"]}</span></div>'
        f'<div class="vb-detail-item"><span class="vb-detail-label">Cuota justa</span><span class="vb-detail-val">{fair}</span></div>'
        f'<div class="vb-detail-item"><span class="vb-detail-label">P. modelo</span><span class="vb-detail-val">{vb["model_p"]*100:.1f}%</span></div>'
        f'<div class="vb-detail-item"><span class="vb-detail-label">P. implícita</span><span class="vb-detail-val">{vb["implied"]*100:.1f}%</span></div>'
        f'<div class="vb-detail-item"><span class="vb-detail-label">Edge</span><span class="vb-detail-val green">+{edge_pp:.1f}pp</span></div>'
        f'<div class="vb-detail-item"><span class="vb-detail-label">xG Local</span><span class="vb-detail-val blue">{xg_h}</span></div>'
        f'<div class="vb-detail-item"><span class="vb-detail-label">xG Visit.</span><span class="vb-detail-val blue">{xg_a}</span></div>'
        f'<div class="vb-detail-item"><span class="vb-detail-label">Kelly 25%</span>'
        f'<span class="vb-detail-val green">🎯 {vb.get("kelly", 0):.1f}% · {round(vb.get("kelly", 0) / 100 * st.session_state.get("bankroll", 100), 1)}u</span></div>'
        + _corner_shot_detail_html(vb) +
        f'</div>'
    )

    hbadge = ctx_badge_html(hctx)
    abadge = ctx_badge_html(actx)

    return (
        f'<div class="vb-card vb-card-{vb["conf_key"]}" style="animation-delay:{delay:.2f}s">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">'
        f'<div>'
        f'<p class="vb-match">{vb["home"]} <span style="color:#94a3b8;font-weight:400">vs</span> {vb["away"]}</p>'
        f'<p class="vb-meta">📅 {date_str} &nbsp;·&nbsp; Jornada {vb.get("matchday","?")} &nbsp;·&nbsp; Mercado: <b>{vb["label"]}</b></p>'
        f'<div class="vb-ctx-row">{hbadge} <span style="color:#cbd5e1;font-size:.75rem;align-self:center">vs</span> {abadge}</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div class="{vb["ev_css"]} ev-pill">📈 EV +{ev_pct:.1f}%</div>'
        f'<div style="margin-top:6px"><span class="conf-tag {vb["conf_css"]}">{vb["conf_icon"]} Confianza {vb["conf_label"]}</span></div>'
        f'</div>'
        f'</div>'
        f'{prob_html}'
        f'{d_row}'
        f'{form_html}'
        f'{alert_html}'
        f'</div>'
    )

# ═══════════════════════════════════════════════════════════
# SIDEBAR SUMMARY (rendered after pre-compute)
# ═══════════════════════════════════════════════════════════
def render_sidebar_summary(all_vb, lc, ev_min_pct):
    n_total  = len(all_vb)
    n_high   = sum(1 for v in all_vb if v["conf_key"]=="high")
    n_medium = sum(1 for v in all_vb if v["conf_key"]=="medium")
    n_low    = sum(1 for v in all_vb if v["conf_key"]=="low")
    avg_ev   = round(sum(v["ev"] for v in all_vb)*100/n_total, 1) if n_total else 0

    # Param bar helper
    def param_bar(label, val_str, pct):
        pct = max(2, min(100, pct))
        return f"""
        <div style="margin:8px 0">
          <div style="display:flex;justify-content:space-between;font-size:.72rem;margin-bottom:3px">
            <span style="color:#64748b">{label}</span>
            <span style="font-weight:700;color:#0f172a">{val_str}</span>
          </div>
          <div style="background:#f1f5f9;border-radius:4px;height:5px">
            <div style="background:linear-gradient(90deg,#00A86B,#0A0D12);width:{pct}%;height:100%;border-radius:4px"></div>
          </div>
        </div>"""

    decay_pct    = int(DECAY_RATE * 1000)        # 0.010 → 10 (out of ~20)
    shrink_pct   = int(SHRINKAGE_K / 20 * 100)  # 10 → 50%
    edge_pct_bar = int(MAX_EDGE * 100 / 25 * 100) # 17/25 → 68%
    ev_bar_pct   = int(ev_min_pct / 15 * 100)   # scaled to 15% max

    with st.sidebar:
        # ── Summary card ──
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0A0D12,#171B22);border-radius:14px;padding:18px;margin:4px 0 16px">
          <div style="color:#3A404A;font-size:.66rem;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace">Picks detectados</div>
          <div style="font-size:2.4rem;font-weight:700;color:white;line-height:1;font-family:'IBM Plex Mono',monospace">{n_total}</div>
          <div style="color:#3A404A;font-size:.70rem;margin-bottom:14px;font-family:'IBM Plex Mono',monospace">value bets · EV medio +{avg_ev}%</div>
          <div style="display:flex;gap:8px">
            <div style="flex:1;background:rgba(0,168,107,0.15);border-radius:8px;padding:8px 6px;text-align:center;border:1px solid rgba(0,168,107,0.28)">
              <div style="color:#00A86B;font-size:1.3rem;font-weight:700;font-family:'IBM Plex Mono',monospace">{n_high}</div>
              <div style="color:#00A86B;font-size:.62rem;margin-top:1px;opacity:.75">Alta</div>
            </div>
            <div style="flex:1;background:rgba(58,64,74,0.20);border-radius:8px;padding:8px 6px;text-align:center;border:1px solid rgba(58,64,74,0.35)">
              <div style="color:#a0aec0;font-size:1.3rem;font-weight:700;font-family:'IBM Plex Mono',monospace">{n_medium}</div>
              <div style="color:#a0aec0;font-size:.62rem;margin-top:1px;opacity:.75">Media</div>
            </div>
            <div style="flex:1;background:rgba(217,119,6,0.14);border-radius:8px;padding:8px 6px;text-align:center;border:1px solid rgba(217,119,6,0.28)">
              <div style="color:#d97706;font-size:1.3rem;font-weight:700;font-family:'IBM Plex Mono',monospace">{n_low}</div>
              <div style="color:#d97706;font-size:.62rem;margin-top:1px;opacity:.75">Baja</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model parameters visual ──
        with st.expander("⚙️ Parámetros del modelo", expanded=False):
            st.markdown(f"""
            <div style="padding:4px 0">
              {param_bar("EV mínimo", f"{ev_min_pct}%", ev_bar_pct)}
              {param_bar("Shrinkage K", str(SHRINKAGE_K), shrink_pct)}
              {param_bar("Decay λ", str(DECAY_RATE), decay_pct * 5)}
              {param_bar("Edge máx.", f"{int(MAX_EDGE*100)}pp", edge_pct_bar)}
            </div>
            <div style="margin-top:8px;font-size:.70rem;color:#94a3b8;line-height:1.5">
              <b>Shrinkage:</b> atrae ratings extremos al promedio cuando la muestra es pequeña.<br>
              <b>Decay:</b> pondera partidos recientes con mayor peso exponencial.<br>
              <b>Edge máx.:</b> filtra picks con ventaja irreal (&gt;{int(MAX_EDGE*100)}pp).
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    # ── Header ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-header">
      <div class="stat-logo">{logo_img(56)}</div>
      <div>
        <div class="stat-title">STATIUM</div>
        <div class="stat-tagline">Sports Intelligence. Predict The Edge.</div>
        <div class="stat-subtitle">Detección de value bets · Modelo Poisson Calibrado · Contexto Competitivo</div>
      </div>
    </div>
    <div class="grad-line"></div>
    """, unsafe_allow_html=True)

    # ── Secrets ──────────────────────────────────────────────
    try:
        FD_KEY   = st.secrets["FOOTBALL_API_KEY"]
        ODDS_KEY = st.secrets["ODDS_API_KEY"]
    except Exception:
        st.error("⚠️ Configura las API Keys en **Settings → Secrets**.")
        st.stop()

    # ── Sidebar – Part 1: Controls ────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 0 14px">
          {logo_img(40)}
          <span style="font-weight:800;font-size:1.05rem;letter-spacing:2px;color:#0A0D12;font-family:'Space Grotesk',sans-serif">STATIUM</span>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown(
            '<div style="background:linear-gradient(135deg,rgba(0,168,107,0.12),rgba(0,168,107,0.03));'
            'border:1px solid rgba(0,168,107,0.30);border-radius:10px;padding:9px 12px;margin-bottom:10px;'
            'display:flex;align-items:center;gap:8px">'
            '<span style="font-size:1.1rem">🌍</span>'
            '<div><div style="font-size:.68rem;font-weight:700;color:#00A86B;letter-spacing:.4px;'
            'font-family:\'IBM Plex Mono\',monospace">EN VIVO · FOCO MUNDIAL 2026</div>'
            '<div style="font-size:.65rem;color:#64748b;margin-top:1px">Las grandes ligas están en receso de temporada</div>'
            '</div></div>',
            unsafe_allow_html=True
        )
        league_opts = list(LEAGUES.keys())
        league_name = st.selectbox(
            "🏆 Liga / Torneo", league_opts, index=0,
            format_func=lambda n: f"{n}  ✨" if LEAGUES[n].get("featured")
                         else f"{n}  · receso" if LEAGUES[n].get("off_season") else n,
        )
        if st.session_state.get("_last_league") != league_name:
            st.session_state["sel_date"] = "all"
            st.session_state["_last_league"] = league_name
        lc = LEAGUES[league_name]
        if lc.get("off_season") and not lc.get("is_tournament"):
            st.markdown(
                '<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;'
                'padding:8px 12px;margin-top:-2px;margin-bottom:10px;font-size:.74rem;color:#92400e;'
                'font-family:\'Space Grotesk\',sans-serif">'
                '😴 Esta liga está fuera de temporada — los datos corresponden a la última campaña '
                'finalizada. Te recomendamos enfocarte en <b>🌍 Mundial 2026</b> para esta semana.'
                '</div>',
                unsafe_allow_html=True
            )
        # Ventana de búsqueda: torneos usan ventana amplia (90 días) para capturar
        # todo el calendario; ligas usan 21 días (filtrado por calendario en la vista)
        days_ahead  = 90 if lc.get("is_tournament") else 21
        ev_min_pct  = st.slider("🎯 EV mínimo (%)", 2, 12, 4)
        ev_threshold = ev_min_pct / 100
        bankroll = st.number_input("💰 Bankroll (u.)", min_value=0.0, value=100.0, step=10.0,
                                   help="Tu bankroll total. Kelly 25% te dirá cuántas unidades apostar.")
        st.session_state["bankroll"] = bankroll
        st.divider()
        if st.button("🔄 Actualizar datos", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # ── Carga ────────────────────────────────────────────────
    is_tournament = lc.get("is_tournament", False)
    with st.spinner("Cargando datos..."):
        if is_tournament:
            season_df  = fetch_intl_matches(FD_KEY)
            hist_mapped = pd.DataFrame()
        else:
            season_df    = fetch_season_matches(FD_KEY, lc["fd"])
            hist_df      = fetch_historical_seasons(lc["fd"], n_seasons=2)
            season_df, hist_mapped = enrich_with_history(season_df, hist_df)
        standings_df = fetch_standings(FD_KEY, lc["fd"])
        upcoming     = fetch_upcoming_matches(FD_KEY, lc["fd"], days_ahead)
        odds_list    = fetch_odds(ODDS_KEY, lc["odds"])

    # Para torneos: combinar fixtures de football-data.org con los derivados de cuotas.
    # No nos quedamos solo con la 1ª fuente que responda — algunas casas de apuestas
    # publican partidos que aún no aparecen en football-data.org (y viceversa), por lo
    # que se fusionan ambas listas evitando duplicados (mismo día + nombres similares).
    if is_tournament and odds_list and not season_df.empty:
        odds_upcoming = upcoming_from_odds(odds_list, season_df, days_ahead=days_ahead)
        if odds_upcoming:
            existing_keys = []
            for m in upcoming:
                try:
                    d = datetime.fromisoformat(m["date"].replace("Z", "+00:00")).date().isoformat()
                except Exception:
                    d = str(m["date"])[:10]
                existing_keys.append((
                    d,
                    m.get("home_name") or "",
                    m.get("away_name") or "",
                ))

            def _is_dup(om):
                try:
                    d = datetime.fromisoformat(om["date"].replace("Z", "+00:00")).date().isoformat()
                except Exception:
                    d = str(om["date"])[:10]
                om_home = om.get("home_name") or ""
                om_away = om.get("away_name") or ""
                for ed, eh, ea in existing_keys:
                    if d != ed:
                        continue
                    # _sim normaliza alias de selecciones (Czechia/Czech Republic,
                    # South Korea/Korea Republic, etc.) antes de comparar
                    h_sim = _sim(om_home, eh)
                    a_sim = _sim(om_away, ea)
                    if h_sim >= 0.62 and a_sim >= 0.62:
                        return True
                return False

            added = [om for om in odds_upcoming if not _is_dup(om)]
            if added:
                upcoming = upcoming + added
                upcoming.sort(key=lambda x: x["date"])
                st.info(
                    f"📅 **{len(added)} partido(s) adicional(es) encontrados vía cuotas** "
                    f"que aún no figuran en football-data.org. Los equipos con historial "
                    f"disponible recibirán predicciones del modelo."
                )

    # ── Info de fuentes usadas (solo torneos) ─────────────────
    if is_tournament and not season_df.empty:
        if "comp_label" in season_df.columns:
            src_counts = season_df["comp_label"].value_counts()
            src_txt = " · ".join(f"**{lbl}** ({cnt})" for lbl, cnt in src_counts.items())
            st.info(f"📡 Fuentes de datos usadas: {src_txt}")

    if season_df.empty:
        if is_tournament:
            st.warning(
                "⚽ **Sin datos históricos disponibles para este torneo.** "
                "Posibles causas: el torneo aún no inició (el Mundial 2026 comienza en junio) "
                "o tu plan de football-data.org no incluye las competiciones internacionales necesarias "
                "(se requiere Tier 2+ para eliminatorias). "
                "El modelo necesita partidos jugados para calcular ratings. "
                "Mientras tanto puedes ver próximos partidos, cuotas y el Tracker."
            )
            ratings, avg_h, avg_a = {}, 1.35, 1.10
        else:
            st.error(
                "❌ No se pudieron cargar partidos históricos. "
                "Verifica que tu **FOOTBALL_API_KEY** esté correctamente configurada en "
                "Settings → Secrets y que tu plan incluya la liga seleccionada."
            )
            st.stop()
    else:
        ratings, avg_h, avg_a = build_ratings(
            season_df,
            decay_rate=INTL_DECAY if is_tournament else DECAY_RATE,
            intl_mode=is_tournament,
        )

    # ── Ratings de corners y tiros (solo ligas, no torneos) ──
    corner_ratings = shot_ratings = sot_ratings = {}
    avg_ch = avg_ca = avg_sh = avg_sa = avg_soth = avg_sota = 5.0
    if not is_tournament and not hist_mapped.empty:
        corner_ratings, avg_ch, avg_ca = build_stat_ratings(hist_mapped, "home_corners", "away_corners")
        shot_ratings,   avg_sh, avg_sa = build_stat_ratings(hist_mapped, "home_shots",   "away_shots")
        sot_ratings, avg_soth, avg_sota = build_stat_ratings(hist_mapped, "home_shots_ot","away_shots_ot")

    # ── Metrics bar ──────────────────────────────────────────
    mc = st.columns(5)
    for col, num, label in zip(mc, [
        len(season_df), len(ratings), len(upcoming),
        len(odds_list), len(standings_df)
    ], ["Partidos históricos","Equipos con rating","Próximos partidos",
        "Mercados con cuotas","Equipos en tabla"]):
        col.markdown(f"""
        <div class="stat-card">
          <div class="stat-card-num">{num}</div>
          <div class="stat-card-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── Pre-compute ──────────────────────────────────────────
    match_map = {}
    all_vb    = []
    for m in upcoming:
        p     = match_probs(m["home_id"], m["away_id"], ratings, avg_h, avg_a)
        om    = find_odds_match(m["home_name"], m["away_name"], m["date"], odds_list)
        bk    = best_odds_for(om)
        md    = int(m["matchday"]) if m["matchday"] else 0
        avg_played = int(standings_df["played"].mean()) if not standings_df.empty else md
        remaining  = max(0, lc["games"] - avg_played)
        td    = is_title_decided(standings_df, remaining)
        hctx  = get_team_context(m["home_id"], standings_df, lc, md, td)
        actx  = get_team_context(m["away_id"], standings_df, lc, md, td)
        alerts= match_alerts(hctx, actx, md, lc, m["home_name"], m["away_name"])
        # Form for card display
        hform = team_form(season_df, m["home_id"], 3)
        aform = team_form(season_df, m["away_id"], 3)
        # Corner / shot predictions
        CORNER_LINES = [8.5, 9.5, 10.5, 11.5]
        SHOT_LINES   = [20.5, 22.5, 24.5]
        SOT_LINES    = [7.5, 8.5, 9.5]
        cp   = stat_ou_probs(m["home_id"], m["away_id"], corner_ratings, avg_ch, avg_ca, CORNER_LINES) if corner_ratings else None
        sp   = stat_ou_probs(m["home_id"], m["away_id"], shot_ratings,   avg_sh, avg_sa, SHOT_LINES)   if shot_ratings   else None
        sotp = stat_ou_probs(m["home_id"], m["away_id"], sot_ratings, avg_soth, avg_sota, SOT_LINES)   if sot_ratings    else None

        vbets = detect_value_bets(p, bk, m["home_name"], m["away_name"], ev_threshold, corner_probs=cp)
        for vb in vbets:
            vb.update({
                "date":     m["date"],
                "matchday": md,
                "home_ctx": hctx,
                "away_ctx": actx,
                "ctx_alerts": alerts,
                "home_id":    m["home_id"],
                "away_id":    m["away_id"],
                "probs":      p,
                "home_form":  hform,
                "away_form":  aform,
                "corner_probs": cp,
                "shot_probs":   sp,
                "sot_probs":    sotp,
            })
            all_vb.append(vb)
        match_map[m["id"]] = {"p":p,"bk":bk,"vbets":vbets,"hctx":hctx,"actx":actx,
                               "alerts":alerts,"md":md,"hform":hform,"aform":aform,
                               "cp":cp,"sp":sp,"sotp":sotp}

    all_vb.sort(key=lambda x: x["ev"], reverse=True)

    # ── Sidebar – Part 2: Summary + Calendario ───────────────
    render_sidebar_summary(all_vb, lc, ev_min_pct)

    # Mapa de fechas con partidos
    match_dates_map: dict = {}
    for m in upcoming:
        d = m["date"][:10]
        match_dates_map[d] = match_dates_map.get(d, 0) + 1
    sorted_match_dates = sorted(match_dates_map.keys())

    # Session state
    if "sel_date" not in st.session_state:
        st.session_state["sel_date"] = "all"
    if "cal_month" not in st.session_state:
        # Mes inicial = mes del primer partido próximo o mes actual
        if sorted_match_dates:
            first_d = datetime.strptime(sorted_match_dates[0], "%Y-%m-%d")
            st.session_state["cal_month"] = (first_d.year, first_d.month)
        else:
            now = datetime.utcnow()
            st.session_state["cal_month"] = (now.year, now.month)

    # Renderizar calendario en sidebar
    with st.sidebar:
        if sorted_match_dates:
            st.divider()
            # ── CSS del calendario: grid uniforme + glow verde en hover/selección ──
            st.markdown(
                """
                <style>
                [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
                    gap: 0.30rem !important;
                    align-items: stretch !important;
                }
                [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"] {
                    min-width: 0 !important;
                    padding: 0 !important;
                    display: flex !important;
                    align-items: stretch !important;
                }
                [data-testid="stSidebar"] .stButton {
                    width: 100% !important;
                    margin: 0 !important;
                }
                [data-testid="stSidebar"] .stButton > button {
                    width: 100% !important;
                    min-height: 32px !important;
                    height: 32px !important;
                    padding: 0 2px !important;
                    margin: 1px 0 !important;
                    font-size: .70rem !important;
                    line-height: 1 !important;
                    border-radius: 8px !important;
                    font-family: 'IBM Plex Mono', monospace !important;
                    transition: transform .15s cubic-bezier(.2,.8,.2,1), box-shadow .18s ease, background .18s ease, color .18s ease, border-color .18s ease !important;
                }
                /* Glow verde al pasar el cursor sobre cualquier día */
                [data-testid="stSidebar"] .stButton > button:hover {
                    background: rgba(0,168,107,0.16) !important;
                    border-color: #00A86B !important;
                    color: #00A86B !important;
                    box-shadow: 0 0 0 2px rgba(0,168,107,0.32), 0 4px 12px rgba(0,168,107,0.20) !important;
                    transform: translateY(-1px) scale(1.04) !important;
                    z-index: 2 !important;
                }
                /* Día seleccionado: glow verde permanente */
                [data-testid="stSidebar"] .stButton > button[kind="primary"],
                [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
                    background: linear-gradient(135deg,#00A86B,#00c97f) !important;
                    border-color: #00A86B !important;
                    color: #ffffff !important;
                    font-weight: 700 !important;
                    box-shadow: 0 0 0 2px rgba(0,168,107,0.45), 0 4px 14px rgba(0,168,107,0.35) !important;
                }
                [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
                [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {
                    box-shadow: 0 0 0 3px rgba(0,168,107,0.6), 0 6px 18px rgba(0,168,107,0.45) !important;
                    transform: translateY(-1px) scale(1.05) !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                '<div style="font-size:.72rem;font-weight:700;color:#64748b;'
                'letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;'
                'font-family:Space Grotesk,sans-serif">📅 Calendario</div>',
                unsafe_allow_html=True
            )

            # Navegación de mes
            cy, cm = st.session_state["cal_month"]
            nav1, nav2, nav3 = st.columns([1, 3, 1])
            if nav1.button("‹", key="cal_prev", use_container_width=True):
                if cm == 1:
                    st.session_state["cal_month"] = (cy - 1, 12)
                else:
                    st.session_state["cal_month"] = (cy, cm - 1)
                st.rerun()
            nav2.markdown(
                f'<div style="text-align:center;font-size:.75rem;font-weight:700;'
                f'color:#0A0D12;font-family:Space Grotesk,sans-serif;padding-top:6px">'
                f'{cal_module.month_name[cm].upper()} {cy}</div>',
                unsafe_allow_html=True
            )
            if nav3.button("›", key="cal_next", use_container_width=True):
                if cm == 12:
                    st.session_state["cal_month"] = (cy + 1, 1)
                else:
                    st.session_state["cal_month"] = (cy, cm + 1)
                st.rerun()

            # Encabezado de días (L M X J V S D)
            hdr_cols = st.columns(7)
            for hc, dname in zip(hdr_cols, ["L","M","X","J","V","S","D"]):
                hc.markdown(
                    f'<div style="text-align:center;font-size:.58rem;color:#94a3b8;'
                    f'font-family:\'IBM Plex Mono\',monospace">{dname}</div>',
                    unsafe_allow_html=True
                )

            # Grid de días — clicables (tap para filtrar por esa fecha)
            today = datetime.utcnow().date()
            sel_d = st.session_state["sel_date"]
            for week in cal_module.monthcalendar(cy, cm):
                wcols = st.columns(7)
                for wc, day in zip(wcols, week):
                    if day == 0:
                        wc.markdown("&nbsp;", unsafe_allow_html=True)
                        continue
                    ds = f"{cy}-{cm:02d}-{day:02d}"
                    n  = match_dates_map.get(ds, 0)
                    is_sel   = ds == sel_d
                    is_today = ds == today.isoformat()
                    if n:
                        cap = f"{day} •" if not is_sel else f"✓ {day}"
                        if wc.button(cap, key=f"cal_day_{ds}", use_container_width=True,
                                     type="primary" if is_sel else "secondary"):
                            st.session_state["sel_date"] = "all" if is_sel else ds
                            st.rerun()
                    else:
                        ring = f"box-shadow:0 0 0 1.5px rgba(26,111,168,0.30);" if is_today else ""
                        wc.markdown(
                            f'<div style="text-align:center;color:#cbd5e1;font-size:.70rem;'
                            f'padding:7px 0;border-radius:6px;{ring}font-family:\'IBM Plex Mono\',monospace">{day}</div>',
                            unsafe_allow_html=True
                        )

            st.markdown(
                '<div style="display:flex;gap:10px;margin:8px 0 2px;flex-wrap:wrap">'
                '<div style="display:flex;align-items:center;gap:4px;font-size:.60rem;color:#64748b">'
                '<div style="width:8px;height:8px;border-radius:2px;border:1px solid rgba(0,168,107,0.45);'
                'background:rgba(0,168,107,0.12)"></div>Con partidos · toca para filtrar</div></div>',
                unsafe_allow_html=True
            )

            # Botón para limpiar selección y ver todos los partidos
            if sel_d != "all":
                if st.button("✕ Ver todos los partidos", key="cal_clear", use_container_width=True):
                    st.session_state["sel_date"] = "all"
                    st.rerun()
                else:
                    st.markdown(
                        f'<div style="text-align:center;font-size:.70rem;color:#00A86B;'
                        f'font-weight:600;margin-top:2px;font-family:\'IBM Plex Mono\',monospace">'
                        f'📅 Filtrando: {datetime.strptime(sel_d,"%Y-%m-%d").strftime("%a %d/%m")}</div>',
                        unsafe_allow_html=True
                    )

    # Aplicar filtro de fecha a la lista de próximos partidos y value bets
    sel_date = st.session_state.get("sel_date", "all")
    if sel_date == "all":
        upcoming_view = upcoming
        all_vb_view   = all_vb
    else:
        upcoming_view = [m for m in upcoming if m["date"][:10] == sel_date]
        all_vb_view   = [v for v in all_vb if v["date"][:10] == sel_date]

    # ── Tabs ─────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs(["🎯 Value Bets","🗓️ Partidos","🔍 Equipo","📋 Clasificación","📈 Tracker"])

    # ─── TAB 1: VALUE BETS ────────────────────────────────────
    with t1:
        st.markdown(f"### 🎯 Value Bets · {league_name}")
        ALL_CONF    = ["Alta","Media","Baja"]
        ALL_MARKETS = ["1 Local","X Empate","2 Visitante","Over 2.5","Under 2.5"]
        col_l, col_r = st.columns([1,1])
        conf_filter   = col_l.multiselect("Confianza", ALL_CONF, default=ALL_CONF,
                                           help="Si lo dejas vacío se muestran todos los niveles de confianza.")
        market_filter = col_r.multiselect("Mercado", ALL_MARKETS, default=ALL_MARKETS,
                                           help="Si lo dejas vacío se muestran todos los mercados.")
        st.markdown("---")

        # Filtro vacío = sin restricción (mostrar todo), no "ocultar todo"
        conf_eff   = conf_filter if conf_filter else ALL_CONF
        market_eff = market_filter if market_filter else ALL_MARKETS
        filtered = [v for v in all_vb_view if v["conf_label"] in conf_eff and v["label"] in market_eff]

        if not filtered:
            st.info("No se detectaron value bets con estos criterios. Prueba bajando el EV mínimo.")
        else:
            col_badge1, col_badge2, col_badge3 = st.columns(3)
            col_badge1.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_GREEN}'>{sum(1 for v in filtered if v['conf_key']=='high')}</div><div class='stat-card-label'>Alta confianza</div></div>", unsafe_allow_html=True)
            col_badge2.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_STEEL}'>{sum(1 for v in filtered if v['conf_key']=='medium')}</div><div class='stat-card-label'>Media confianza</div></div>", unsafe_allow_html=True)
            col_badge3.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_AMBER}'>{sum(1 for v in filtered if v['conf_key']=='low')}</div><div class='stat-card-label'>Baja confianza</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

            # ── Agrupar picks por partido (orden = mejor EV primero) ──
            from collections import OrderedDict
            groups = OrderedDict()
            for vb in filtered:
                gkey = (vb["home"], vb["away"], vb["date"])
                groups.setdefault(gkey, []).append(vb)

            for gi, ((g_home, g_away, g_date), vbs) in enumerate(groups.items()):
                try:
                    g_dt_str = fmt_match_dt(g_date)
                except Exception:
                    g_dt_str = g_date[:16]

                st.markdown(
                    f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
                    f'flex-wrap:wrap;gap:8px;margin:{"6" if gi==0 else "22"}px 0 10px">'
                    f'<div style="font-size:1.05rem;font-weight:700;color:{BRAND_DARK};'
                    f'font-family:\'Space Grotesk\',sans-serif">⚽ {g_home} <span style="color:#94a3b8;font-weight:400">vs</span> {g_away}</div>'
                    f'<div style="font-size:.74rem;color:#94a3b8;font-family:\'IBM Plex Mono\',monospace">'
                    f'📅 {g_dt_str} &nbsp;·&nbsp; {len(vbs)} pick(s) detectado(s)</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                col_picks, col_why = st.columns([3, 2])
                with col_picks:
                    for idx, vb in enumerate(vbs):
                        st.markdown(vb_card_html(vb, idx), unsafe_allow_html=True)
                with col_why:
                    st.markdown(
                        '<div style="font-size:.66rem;font-weight:700;color:#94a3b8;letter-spacing:.8px;'
                        'text-transform:uppercase;margin:4px 0 8px;'
                        'font-family:\'IBM Plex Mono\',monospace">📊 ¿Por qué estos picks?</div>',
                        unsafe_allow_html=True
                    )
                    for idx, vb in enumerate(vbs):
                        with st.expander(f"{vb['conf_icon']} {vb['label']} · cuota {vb['bk_odds']} · EV +{vb['ev']*100:.1f}%",
                                         expanded=(idx == 0)):
                            analysis = generate_analysis(vb, ratings)
                            st.markdown(analysis)
                st.markdown('<div class="grad-line" style="margin:14px 0 4px;opacity:.5"></div>', unsafe_allow_html=True)

            st.markdown("---")
            df_vb = pd.DataFrame(filtered)
            df_vb["EV %"]     = (df_vb["ev"]*100).round(1)
            df_vb["Edge pp"]  = (df_vb["edge"]*100).round(1)
            df_vb["Modelo %"] = (df_vb["model_p"]*100).round(1)
            df_vb["Impl. %"]  = (df_vb["implied"]*100).round(1)
            df_vb["Ctx L"]    = df_vb["home_ctx"].apply(lambda x:f"{x['emoji']} {x['label']}")
            df_vb["Ctx V"]    = df_vb["away_ctx"].apply(lambda x:f"{x['emoji']} {x['label']}")
            show=["home","away","label","bk_odds","Modelo %","Impl. %","Edge pp","EV %","conf_label","Ctx L","Ctx V"]
            st.dataframe(df_vb[show].rename(columns={"home":"Local","away":"Visitante","label":"Mercado","bk_odds":"Cuota","conf_label":"Conf."}), use_container_width=True, hide_index=True)
            csv=df_vb[show].to_csv(index=False)
            st.download_button("📥 Exportar CSV", csv, f"statitum_{datetime.now().strftime('%Y%m%d')}.csv","text/csv",use_container_width=True)

    # ─── TAB 2: PARTIDOS ──────────────────────────────────────
    with t2:
        date_lbl = f"· {sel_date}" if sel_date != "all" else ""
        st.markdown(f"### 🗓️ Partidos · {league_name} {date_lbl}")
        if not upcoming_view:
            st.info("No hay partidos en este período. Selecciona otra fecha o pulsa 'Todos'.")
        for m in upcoming_view:
            info  = match_map[m["id"]]
            p,bk  = info["p"], info["bk"]
            hctx,actx,alerts = info["hctx"],info["actx"],info["alerts"]
            suffix = f"  🎯 {len(info['vbets'])}" if info["vbets"] else ""
            suffix += "  ⚠️" if alerts else ""
            with st.expander(
                f"{hctx['emoji']} **{m['home_name']}** vs **{m['away_name']}** {actx['emoji']} "
                f"· {fmt_match_dt(m['date'])}{suffix}", expanded=False):
                col1,col2=st.columns(2)
                col1.markdown(f"{ctx_badge_html(hctx)}", unsafe_allow_html=True)
                col2.markdown(f"{ctx_badge_html(actx)}", unsafe_allow_html=True)
                for a in alerts: st.warning(a)
                if not p:
                    st.info("Datos insuficientes.")
                    continue
                fh=team_form(season_df,m["home_id"]); fa=team_form(season_df,m["away_id"])
                c1,c2,c3=st.columns([5,1,5])
                c1.markdown(f"**🏠 {m['home_name']}**<br>{render_form(fh)}", unsafe_allow_html=True)
                c2.markdown("**VS**")
                c3.markdown(f"**✈️ {m['away_name']}**<br>{render_form(fa)}", unsafe_allow_html=True)
                st.markdown(prob_bar_html(p,m["home_name"],m["away_name"]),unsafe_allow_html=True)
                if bk["h2h_1"]>0:
                    st.divider()
                    b1,b2,b3,b4,b5=st.columns(5)
                    b1.metric("🏠 Local",    bk["h2h_1"],f"Justa {p['fair_1']}")
                    b2.metric("🤝 Empate",   bk["h2h_x"],f"Justa {p['fair_x']}")
                    b3.metric("✈️ Visit.",   bk["h2h_2"],f"Justa {p['fair_2']}")
                    b4.metric("Over 2.5",    bk["o25"],  f"Justa {p['fair_o25']}")
                    b5.metric("Under 2.5",   bk["u25"],  f"Justa {p['fair_u25']}")
                if info["vbets"]:
                    for vb in info["vbets"]:
                        st.success(f"🎯 **{vb['label']}** @ {vb['bk_odds']} · Modelo {vb['model_p']*100:.1f}% · Implícita {vb['implied']*100:.1f}% · **EV +{vb['ev']*100:.1f}%** · {vb['conf_icon']} {vb['conf_label']}")
                else:
                    sugg = suggest_top_market(p, bk)
                    if sugg:
                        st.info(
                            f"💡 **Sugerencia del modelo (sin edge de valor):** "
                            f"**{sugg['label']}** @ {sugg['bk_odds']} · Modelo {sugg['model_p']*100:.1f}% "
                            f"· Implícita {sugg['implied']*100:.1f}% · cuota justa ≈ {sugg['fair']} — "
                            f"el mercado más probable según el modelo, aunque la cuota actual no ofrece valor (EV) suficiente."
                        )
                    else:
                        st.caption("Sin cuotas disponibles para sugerir un mercado en este partido todavía.")

    # ─── TAB 3: EQUIPO ────────────────────────────────────────
    with t3:
        st.markdown("### 🔍 Análisis de equipo")
        all_t={row["home_id"]:row["home_name"] for _,row in season_df.iterrows()}
        all_t.update({row["away_id"]:row["away_name"] for _,row in season_df.iterrows()})
        n2id={v:k for k,v in all_t.items()}
        sel=st.selectbox("Equipo",sorted(all_t.values()))
        sid=n2id[sel]
        if not standings_df.empty:
            lmd=int(standings_df["played"].max())
            td=is_title_decided(standings_df,max(0,lc["games"]-lmd))
            sctx=get_team_context(sid,standings_df,lc,lmd,td)
            st.markdown(f'<div style="margin:8px 0">{ctx_badge_html(sctx)}</div>',unsafe_allow_html=True)
        if sid in ratings:
            r=ratings[sid]
            m1,m2,m3,m4,m5=st.columns(5)
            m1.metric("Partidos",r["n"]); m2.metric("N efectivo",r["n_eff"])
            m3.metric("Goles/pj",r["gs_avg"]); m4.metric("Recibidos/pj",r["gc_avg"])
            m5.metric("Diferencia",f"{round(r['gs_avg']-r['gc_avg'],2):+.2f}")
            st.markdown("**Forma reciente**")
            f=team_form(season_df,sid,8)
            if f: st.markdown(render_form(f),unsafe_allow_html=True)
            st.divider()
            st.markdown("**Ratings calibrados (1.00 = media de liga)**")
            r1,r2,r3,r4=st.columns(4)
            r1.metric("Ataque local",   round(r["att_h"],3),"↑" if r["att_h"]>1 else "↓")
            r2.metric("Ataque visitante",round(r["att_a"],3),"↑" if r["att_a"]>1 else "↓")
            r3.metric("Defensa local",  round(r["def_h"],3),"↓ sólida" if r["def_h"]<1 else "↑ porosa")
            r4.metric("Def. visitante", round(r["def_a"],3),"↓ sólida" if r["def_a"]<1 else "↑ porosa")
            hm_t=season_df[season_df["home_id"]==sid][["date","home_goals","away_goals"]].rename(columns={"home_goals":"gf","away_goals":"gc"})
            am_t=season_df[season_df["away_id"]==sid][["date","home_goals","away_goals"]].rename(columns={"away_goals":"gf","home_goals":"gc"})
            all_m=pd.concat([hm_t,am_t]).sort_values("date").reset_index(drop=True)
            if not all_m.empty:
                import plotly.graph_objects as go
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=all_m.index,y=all_m["gf"],name="Anotados",
                    line=dict(color=BRAND_GREEN,width=2.5),mode="lines+markers",
                    marker=dict(size=5)))
                fig.add_trace(go.Scatter(x=all_m.index,y=all_m["gc"],name="Recibidos",
                    line=dict(color="#ef4444",width=2.5),mode="lines+markers",
                    marker=dict(size=5)))
                fig.add_hline(y=r["gs_avg"],line_dash="dot",line_color=BRAND_GREEN,opacity=0.6)
                fig.add_hline(y=r["gc_avg"],line_dash="dot",line_color="#ef4444",opacity=0.6)
                fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h"),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(248,250,252,1)",
                    xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig,use_container_width=True)
        else:
            st.warning("Datos insuficientes para este equipo.")

    # ─── TAB 4: CLASIFICACIÓN ─────────────────────────────────
    with t4:
        st.markdown(f"### 📋 Clasificación · {league_name}")
        if standings_df.empty:
            st.warning("No disponible.")
        else:
            lmd=int(standings_df["played"].max())
            td=is_title_decided(standings_df,max(0,lc["games"]-lmd))
            def ctx_row(row):
                c=get_team_context(row["team_id"],standings_df,lc,lmd,td)
                return f"{c['emoji']} {c['label']}"
            sdf=standings_df.copy()
            sdf["Situación"]=sdf.apply(ctx_row,axis=1)
            sdf["gd"]=sdf["gd"].apply(lambda x:f"{int(x):+d}")
            st.dataframe(
                sdf.rename(columns={"position":"Pos","team_name":"Equipo","played":"PJ",
                    "points":"Pts","gf":"GF","ga":"GC","gd":"DG"})
                [["Pos","Equipo","PJ","Pts","GF","GC","DG","Situación"]],
                use_container_width=True, hide_index=True)

    # ─── TAB 5: TRACKER ──────────────────────────────────────
    with t5:
        st.markdown("### 📈 Tracker · Racha & Historial")
        history = load_history()
        streak_count, streak_type = get_streak(history)
        stats = get_stats(history)
        pending_picks = [p for p in history if p["result"] == "pending"]
        resolved_picks = sorted(
            [p for p in history if p["result"] in ("hit","miss")],
            key=lambda x: x.get("resolved_at") or "", reverse=True
        )

        # ── Stats row ──────────────────────────────────────
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        streak_color = BRAND_GREEN if streak_type == "hit" else ("#ef4444" if streak_type == "miss" else "#94a3b8")
        sc1.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{streak_color}'>{streak_count}</div><div class='stat-card-label'>Racha activa</div></div>", unsafe_allow_html=True)
        sc2.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_GREEN}'>{stats['win_rate']}%</div><div class='stat-card-label'>Win rate · {stats['hits']}H / {stats['misses']}M</div></div>", unsafe_allow_html=True)
        roi_color = BRAND_GREEN if stats['roi'] >= 0 else "#ef4444"
        sc3.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{roi_color}'>{stats['roi']:+.1f}%</div><div class='stat-card-label'>ROI</div></div>", unsafe_allow_html=True)
        pnl_color = BRAND_GREEN if stats['pnl'] >= 0 else "#ef4444"
        sc4.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{pnl_color}'>{stats['pnl']:+.2f}</div><div class='stat-card-label'>P&L · Apostado {stats['staked']:.2f}</div></div>", unsafe_allow_html=True)
        if stats.get("avg_clv") is not None:
            clv_color = BRAND_GREEN if stats["avg_clv"] >= 0 else "#ef4444"
            sc5.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{clv_color}'>{stats['avg_clv']:+.1f}%</div><div class='stat-card-label'>CLV medio · {stats['n_clv']} picks</div></div>", unsafe_allow_html=True)
        else:
            sc5.markdown("<div class='stat-card'><div class='stat-card-num' style='color:#94a3b8'>—</div><div class='stat-card-label'>CLV medio</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

        # ── Streak visual + Calendario ─────────────────────
        col_streak, col_cal = st.columns([1, 2])

        with col_streak:
            st.markdown(streak_html(streak_count, streak_type), unsafe_allow_html=True)

        with col_cal:
            now = datetime.utcnow()
            cal_month = st.selectbox(
                "Mes",
                options=[(now.year, now.month),
                         (now.year, now.month-1) if now.month > 1 else (now.year-1, 12),
                         (now.year, now.month-2) if now.month > 2 else (now.year-1, 14-now.month)],
                format_func=lambda ym: f"{cal_module.month_name[ym[1]]} {ym[0]}",
                label_visibility="collapsed",
            )
            st.markdown(calendar_html_grid(history, cal_month[0], cal_month[1]), unsafe_allow_html=True)

        st.markdown("---")

        # ── Registrar pick ────────────────────────────────
        with st.expander("➕ Registrar pick", expanded=(len(history)==0)):
            # Opciones: picks activos detectados + manual
            pick_options = {"✏️ Entrada manual": None}
            for vb in all_vb:
                key = f"🎯 {vb['home']} vs {vb['away']} · {vb['label']} @ {vb['bk_odds']}"
                pick_options[key] = vb

            with st.form("form_add_pick", clear_on_submit=True):
                sel_label = st.selectbox("Pick", list(pick_options.keys()))
                sel_vb    = pick_options[sel_label]

                fc1, fc2 = st.columns(2)
                home_val  = fc1.text_input("Local",   value=sel_vb["home"]  if sel_vb else "")
                away_val  = fc2.text_input("Visitante", value=sel_vb["away"] if sel_vb else "")
                fc3, fc4  = st.columns(2)
                league_val = fc3.text_input("Liga", value=league_name)
                market_val = fc4.selectbox("Mercado",
                    ["1 Local","X Empate","2 Visitante","Over 2.5","Under 2.5"],
                    index=["1 Local","X Empate","2 Visitante","Over 2.5","Under 2.5"].index(sel_vb["label"]) if sel_vb else 0)
                fc5, fc6, fc7 = st.columns(3)
                odds_val   = fc5.number_input("Cuota entrada", min_value=1.01, max_value=20.0,
                                              value=float(sel_vb["bk_odds"]) if sel_vb else 2.0, step=0.05)
                stake_val  = fc6.number_input("Stake (u.)", min_value=0.0, value=10.0, step=1.0)
                result_val = fc7.selectbox("Resultado", ["Pendiente","Hit ✅","Miss ❌"])
                fc8, fc9 = st.columns(2)
                closing_val = fc8.number_input("Cuota cierre (opcional)",
                                               min_value=0.0, max_value=20.0, value=0.0, step=0.05,
                                               help="Cuota justo antes del partido. Permite calcular CLV.")
                date_val   = fc9.date_input("Fecha del partido",
                                            value=datetime.fromisoformat(sel_vb["date"].replace("Z","+00:00")).date() if sel_vb else datetime.utcnow().date())

                submitted = st.form_submit_button("💾 Guardar pick", use_container_width=True)
                if submitted:
                    if not home_val or not away_val:
                        st.error("Completa Local y Visitante.")
                    else:
                        result_map = {"Pendiente":"pending","Hit ✅":"hit","Miss ❌":"miss"}
                        model_p_val = sel_vb["model_p"] if sel_vb else 0.5
                        ev_val      = sel_vb["ev"]      if sel_vb else 0.0
                        co = closing_val if closing_val > 1.0 else None
                        add_pick_to_history(
                            home=home_val, away=away_val, league=league_val,
                            market=market_val, odds=odds_val,
                            model_p=model_p_val, ev=ev_val,
                            stake=stake_val,
                            date_match=date_val.isoformat(),
                            closing_odds=co,
                        )
                        r = result_map[result_val]
                        if r != "pending":
                            new_hist = load_history()
                            if new_hist:
                                resolve_pick(new_hist[-1]["id"], r, stake_val, closing_odds=co)
                        st.success(f"✅ Pick guardado: {home_val} vs {away_val} · {market_val} @ {odds_val}")
                        st.rerun()

        # ── Pendientes ────────────────────────────────────
        if pending_picks:
            st.markdown(f"#### ⏳ Pendientes de resolución ({len(pending_picks)})")
            for p in sorted(pending_picks, key=lambda x: x.get("date_match",""), reverse=True):
                with st.container():
                    pc1, pc_co, pc2, pc3, pc4 = st.columns([3, 1.2, 0.9, 0.9, 0.6])
                    pc1.markdown(
                        f"**{p['home']} vs {p['away']}**  \n"
                        f"<span style='font-size:.75rem;color:#64748b;font-family:IBM Plex Mono'>"
                        f"{p['market']} @ {p['odds']} · EV +{p['ev']}% · Stake {p['stake']}</span>",
                        unsafe_allow_html=True
                    )
                    close_odds_val = pc_co.number_input(
                        "Cuota cierre", min_value=0.0, max_value=20.0, value=0.0, step=0.05,
                        key=f"co_{p['id']}", label_visibility="collapsed",
                        help="Cuota de cierre para calcular CLV (opcional)",
                    )
                    co_input = close_odds_val if close_odds_val > 1.0 else None
                    if pc2.button("✅ Hit",  key=f"hit_{p['id']}"):
                        resolve_pick(p["id"], "hit",  p["stake"], closing_odds=co_input); st.rerun()
                    if pc3.button("❌ Miss", key=f"miss_{p['id']}"):
                        resolve_pick(p["id"], "miss", p["stake"], closing_odds=co_input); st.rerun()
                    if pc4.button("🗑️",     key=f"del_{p['id']}"):
                        delete_pick(p["id"]); st.rerun()
            st.markdown("---")

        # ── Historial resuelto ────────────────────────────
        if resolved_picks:
            st.markdown(f"#### 📋 Historial ({len(resolved_picks)} picks resueltos)")
            df_hist = pd.DataFrame(resolved_picks)
            df_hist["Resultado"] = df_hist["result"].map({"hit":"✅ Hit","miss":"❌ Miss"})
            df_hist["P&L"]       = df_hist["pnl"].apply(lambda x: f"{x:+.2f}" if x is not None else "—")
            df_hist["ROI pick"]  = df_hist.apply(
                lambda r: f"{(r['pnl']/r['stake']*100):+.1f}%" if r.get('stake') and r['stake']>0 else "—", axis=1)
            df_hist["CLV"]       = df_hist.apply(
                lambda r: f"{r['clv']:+.1f}%" if r.get("clv") is not None else "—", axis=1)
            show_cols = ["home","away","league","market","odds","model_p","ev","stake","Resultado","P&L","ROI pick","CLV"]
            rename    = {"home":"Local","away":"Visitante","league":"Liga","market":"Mercado",
                         "odds":"Cuota","model_p":"P.Modelo%","ev":"EV%","stake":"Stake"}
            st.dataframe(
                df_hist[show_cols].rename(columns=rename),
                use_container_width=True, hide_index=True
            )
            csv_hist = df_hist[show_cols].rename(columns=rename).to_csv(index=False)
            st.download_button("📥 Exportar historial CSV", csv_hist,
                               f"statium_historial_{datetime.now().strftime('%Y%m%d')}.csv",
                               "text/csv", use_container_width=True)
        elif not pending_picks:
            st.info("Aún no hay picks registrados. Usa '➕ Registrar pick' para empezar a trackear.")

        # ── Tabla de precisión por fecha ──────────────────────
        all_resolved = [p for p in history if p["result"] in ("hit", "miss")]
        if all_resolved:
            st.markdown("---")
            st.markdown("#### 🎯 Precisión del modelo por fecha de partido")

            # Agrupar por date_match
            from collections import defaultdict
            by_date = defaultdict(list)
            for p in all_resolved:
                by_date[p.get("date_match", "Sin fecha")].append(p)

            prec_rows = []
            for date_str in sorted(by_date.keys(), reverse=True):
                picks_day = by_date[date_str]
                hits   = sum(1 for p in picks_day if p["result"] == "hit")
                misses = sum(1 for p in picks_day if p["result"] == "miss")
                total  = hits + misses
                wr     = round(hits / total * 100, 1) if total else 0
                pnl_d  = sum((p.get("pnl") or 0) for p in picks_day)
                avg_ev = round(sum(p.get("ev", 0) for p in picks_day) / total, 1) if total else 0
                avg_odds = round(sum(p.get("odds", 0) for p in picks_day) / total, 2) if total else 0
                prec_rows.append({
                    "Fecha":     date_str,
                    "Picks":     total,
                    "✅ Hits":   hits,
                    "❌ Misses": misses,
                    "Win Rate":  f"{wr}%",
                    "P&L día":   f"{pnl_d:+.2f}",
                    "EV medio":  f"+{avg_ev}%",
                    "Cuota media": avg_odds,
                })

            df_prec = pd.DataFrame(prec_rows)

            # Colorear win rate con st.dataframe styler
            def color_wr(val):
                try:
                    v = float(val.replace("%",""))
                    if v >= 60:   return "color: #00A86B; font-weight: 700"
                    elif v >= 45: return "color: #d97706; font-weight: 600"
                    else:         return "color: #ef4444; font-weight: 600"
                except Exception:
                    return ""

            def color_pnl(val):
                try:
                    v = float(val)
                    return "color: #00A86B; font-weight:700" if v >= 0 else "color: #ef4444; font-weight:700"
                except Exception:
                    return ""

            styled = (
                df_prec.style
                .applymap(color_wr,  subset=["Win Rate"])
                .applymap(color_pnl, subset=["P&L día"])
            )

            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Resumen acumulado al pie
            total_all  = len(all_resolved)
            hits_all   = sum(1 for p in all_resolved if p["result"] == "hit")
            wr_all     = round(hits_all / total_all * 100, 1) if total_all else 0
            pnl_all    = sum((p.get("pnl") or 0) for p in all_resolved)
            best_day   = max(prec_rows, key=lambda r: float(r["P&L día"]))
            worst_day  = min(prec_rows, key=lambda r: float(r["P&L día"]))

            ba1, ba2, ba3, ba4 = st.columns(4)
            ba1.metric("Win Rate total", f"{wr_all}%",   f"{hits_all}H / {total_all - hits_all}M")
            ba2.metric("P&L acumulado",  f"{pnl_all:+.2f}")
            ba3.metric("Mejor fecha",    best_day["Fecha"],  best_day["P&L día"])
            ba4.metric("Peor fecha",     worst_day["Fecha"], worst_day["P&L día"])

            st.download_button(
                "📥 Exportar tabla de precisión CSV",
                df_prec.to_csv(index=False),
                f"statium_precision_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv", use_container_width=True
            )

    st.markdown("""
    <div class="footer">
      <b>STATIUM</b> · Sports Intelligence. Predict The Edge.<br>
      Poisson Calibrado · Shrinkage Bayesiano · Contexto Competitivo<br>
      Datos: football-data.org · Cuotas: the-odds-api.com · Las probabilidades son estimaciones estadísticas.
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()