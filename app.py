import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import warnings, json, os, uuid, calendar as cal_module
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

  /* ── Header ── */
  .stat-header {{
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 8px 0;
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

LEAGUES = {
    "🇬🇧 Premier League": {"fd":"PL",  "odds":"soccer_epl",               "games":38,"teams":20,"cl":4,"euro":6,"rel":3},
    "🇪🇸 La Liga":         {"fd":"PD",  "odds":"soccer_spain_la_liga",      "games":38,"teams":20,"cl":4,"euro":7,"rel":3},
    "🇮🇹 Serie A":         {"fd":"SA",  "odds":"soccer_italy_serie_a",      "games":38,"teams":20,"cl":4,"euro":7,"rel":3},
    "🇩🇪 Bundesliga":      {"fd":"BL1", "odds":"soccer_germany_bundesliga", "games":34,"teams":18,"cl":4,"euro":6,"rel":2},
    "🇫🇷 Ligue 1":         {"fd":"FL1", "odds":"soccer_france_ligue_one",   "games":34,"teams":18,"cl":3,"euro":5,"rel":3},
}

MIN_ODDS     = 1.35
MAX_ODDS     = 7.00
MAX_EDGE     = 0.17
SHRINKAGE_K  = 10
DECAY_RATE   = 0.010
LATE_SEASON  = 5

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picks_history.json")

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
    return [{"id":m["id"],"date":m["utcDate"],
             "home_id":m["homeTeam"]["id"],"home_name":m["homeTeam"]["name"],
             "away_id":m["awayTeam"]["id"],"away_name":m["awayTeam"]["name"],
             "matchday":m.get("matchday") or 0} for m in data.get("matches",[])]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_standings(fd_key, fd_code):
    data = _fd_get(fd_key, f"/competitions/{fd_code}/standings")
    if not data: return pd.DataFrame()
    for s in data.get("standings",[]):
        if s.get("type") == "TOTAL":
            return pd.DataFrame([{
                "position":e["position"],"team_id":e["team"]["id"],
                "team_name":e["team"]["name"],"played":e["playedGames"],
                "points":e["points"],"gf":e["goalsFor"],
                "ga":e["goalsAgainst"],"gd":e["goalDifference"],
            } for e in s.get("table",[])])
    return pd.DataFrame()

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_odds(odds_key, sport_key):
    data = _odds_get(odds_key, f"/sports/{sport_key}/odds/",
                     {"regions":"eu","markets":"h2h,totals","oddsFormat":"decimal"})
    return data if isinstance(data, list) else []

# ═══════════════════════════════════════════════════════════
# CONTEXTO COMPETITIVO
# ═══════════════════════════════════════════════════════════
def is_title_decided(standings_df, remaining):
    if standings_df.empty or len(standings_df) < 2:
        return False
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
    row       = row.iloc[0]
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

def match_alerts(home_ctx, away_ctx, matchday, league_cfg):
    alerts = []
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
def build_ratings(df):
    if df.empty or len(df)<20: return {}, 1.35, 1.10
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["days_ago"] = (now - df["date_parsed"]).dt.days.clip(lower=0)
    df["w"] = np.exp(-DECAY_RATE * df["days_ago"])
    avg_h = float(np.average(df["home_goals"].values, weights=df["w"].values))
    avg_a = float(np.average(df["away_goals"].values, weights=df["w"].values))
    if avg_h==0 or avg_a==0: return {}, 1.35, 1.10
    mean_w = df["w"].mean()
    def wavg(vals, wts, fb=1.0):
        s=wts.sum(); return float(np.average(vals,weights=wts)) if (len(vals)>=1 and s>0) else fb
    def shrink(raw, n):
        return (n*raw+SHRINKAGE_K*1.0)/(n+SHRINKAGE_K)
    ratings={}
    for tid in set(df["home_id"])|set(df["away_id"]):
        hm=df[df["home_id"]==tid]; am=df[df["away_id"]==tid]
        nh,na=len(hm),len(am)
        if nh+na<4: continue
        nh_eff=hm["w"].sum()/mean_w if nh>=1 else 0.0
        na_eff=am["w"].sum()/mean_w if na>=1 else 0.0
        att_h=shrink(wavg(hm["home_goals"].values,hm["w"].values)/avg_h if nh>=2 else 1.0, nh_eff)
        att_a=shrink(wavg(am["away_goals"].values,am["w"].values)/avg_a if na>=2 else 1.0, na_eff)
        def_h=shrink(wavg(hm["away_goals"].values,hm["w"].values)/avg_a if nh>=2 else 1.0, nh_eff)
        def_a=shrink(wavg(am["home_goals"].values,am["w"].values)/avg_h if na>=2 else 1.0, na_eff)
        gs_avg=(hm["home_goals"].sum()+am["away_goals"].sum())/(nh+na)
        gc_avg=(hm["away_goals"].sum()+am["home_goals"].sum())/(nh+na)
        ratings[tid]={"att_h":round(att_h,3),"att_a":round(att_a,3),
                      "def_h":round(def_h,3),"def_a":round(def_a,3),
                      "gs_avg":round(gs_avg,2),"gc_avg":round(gc_avg,2),
                      "n":nh+na,"n_eff":round(nh_eff+na_eff,1)}
    return ratings, avg_h, avg_a

def match_probs(home_id, away_id, ratings, avg_h, avg_a):
    if home_id not in ratings or away_id not in ratings: return None
    hr,ar = ratings[home_id], ratings[away_id]
    lam_h=float(np.clip(hr["att_h"]*ar["def_a"]*avg_h,0.40,4.5))
    lam_a=float(np.clip(ar["att_a"]*hr["def_h"]*avg_a,0.40,4.5))
    G=8
    M=np.outer([poisson.pmf(i,lam_h) for i in range(G)],[poisson.pmf(i,lam_a) for i in range(G)])
    hw=float(np.sum(np.tril(M,-1))); dr=float(np.sum(np.diag(M))); aw=float(np.sum(np.triu(M,1)))
    o25=float(sum(M[i][j] for i in range(G) for j in range(G) if i+j>2))
    btts=float((1-poisson.pmf(0,lam_h))*(1-poisson.pmf(0,lam_a)))
    f=lambda p: round(1/p,2) if p>0.01 else 99.0
    return {"lam_h":round(lam_h,2),"lam_a":round(lam_a,2),
            "home_win":round(hw,4),"draw":round(dr,4),"away_win":round(aw,4),
            "over25":round(o25,4),"under25":round(1-o25,4),"btts":round(btts,4),
            "fair_1":f(hw),"fair_x":f(dr),"fair_2":f(aw),
            "fair_o25":f(o25),"fair_u25":f(1-o25),"fair_btts":f(btts)}

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
def _sim(a,b):
    for s in [" fc"," cf"," afc"," sc"," united"]:
        a=a.lower().replace(s,""); b=b.lower().replace(s,"")
    return SequenceMatcher(None,a.strip(),b.strip()).ratio()

def find_odds_match(fd_home, fd_away, fd_date_str, odds_list):
    fd_date=datetime.fromisoformat(fd_date_str.replace("Z","+00:00")).date()
    best,bs=None,0
    for om in odds_list:
        try: om_date=datetime.fromisoformat(om["commence_time"].replace("Z","+00:00")).date()
        except: continue
        if abs((om_date-fd_date).days)>1: continue
        score=_sim(fd_home,om["home_team"])+_sim(fd_away,om["away_team"])
        if score>bs and score>1.25: bs,best=score,om
    return best

def best_odds_for(om):
    best={"h2h_1":0,"h2h_x":0,"h2h_2":0,"o25":0,"u25":0}
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
                    if abs(pt-2.5)<0.01:
                        if oc.get("name","").lower()=="over": best["o25"]=max(best["o25"],pr)
                        else: best["u25"]=max(best["u25"],pr)
    return best

def conf_info(edge):
    if   edge<=0.08: return "Alta",  "high",  "🟢","ev-high","conf-high"
    elif edge<=0.13: return "Media", "medium","🟡","ev-medium","conf-medium"
    else:            return "Baja",  "low",   "🟠","ev-low","conf-low"

def detect_value_bets(probs, bk, home_name, away_name, ev_threshold):
    if not probs: return []
    checks=[("1 Local",probs["home_win"],bk["h2h_1"]),
            ("X Empate",probs["draw"],bk["h2h_x"]),
            ("2 Visitante",probs["away_win"],bk["h2h_2"]),
            ("Over 2.5",probs["over25"],bk["o25"]),
            ("Under 2.5",probs["under25"],bk["u25"])]
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
                      "home":home_name,"away":away_name})
    return found

# ═══════════════════════════════════════════════════════════
# TRACKER — PERSISTENCIA Y ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════
def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def _save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def add_pick_to_history(home, away, league, market, odds, model_p, ev, stake, date_match):
    history = load_history()
    pick = {
        "id":         str(uuid.uuid4())[:8],
        "created_at": datetime.utcnow().isoformat(),
        "date_match": date_match,
        "home": home, "away": away,
        "league": league, "market": market,
        "odds":    round(float(odds), 2),
        "model_p": round(float(model_p)*100, 1),
        "ev":      round(float(ev)*100, 1),
        "stake":   round(float(stake), 2) if stake else 0.0,
        "result":  "pending",
        "resolved_at": None, "pnl": None,
    }
    history.append(pick)
    _save_history(history)
    return pick["id"]

def resolve_pick(pick_id, result, stake=None):
    history = load_history()
    for p in history:
        if p["id"] == pick_id:
            p["result"] = result
            p["resolved_at"] = datetime.utcnow().isoformat()
            if stake is not None: p["stake"] = round(float(stake), 2)
            if result == "hit":   p["pnl"] = round(p["stake"] * (p["odds"] - 1), 2)
            elif result == "miss": p["pnl"] = -p["stake"]
            else:                  p["pnl"] = 0.0
            break
    _save_history(history)

def delete_pick(pick_id):
    history = [p for p in load_history() if p["id"] != pick_id]
    _save_history(history)

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
    return {
        "total":    len(resolved),
        "hits":     hits,
        "misses":   len(resolved)-hits,
        "win_rate": round(hits/len(resolved)*100, 1),
        "roi":      round(pnl/staked*100, 1) if staked > 0 else 0.0,
        "pnl":      round(pnl, 2),
        "staked":   round(staked, 2),
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

    def sample_note(n_eff):
        if n_eff < 7:  return f"⚠️ muestra pequeña ({n_eff:.1f} partidos efectivos) — rating fuertemente ajustado al promedio."
        elif n_eff < 15: return f"muestra moderada ({n_eff:.1f} partidos efectivos)."
        else:           return f"muestra sólida ({n_eff:.1f} partidos efectivos)."

    lines = []

    # ── Market-specific reasoning ──────────────────────────
    if label == "1 Local":
        lines.append(f"#### 🏠 Análisis: Victoria de {home}")
        lines.append(
            f"**Ataque local de {home}:** `{att_h:.3f}x` la media de liga — {rating_desc(att_h)}. "
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
        f"**🔬 Calidad de datos:** {home} — {sample_note(n_h)}  "
        f"{away} — {sample_note(n_a)}"
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

def vb_card_html(vb, idx=0):
    dt       = datetime.fromisoformat(vb["date"].replace("Z","+00:00"))
    date_str = dt.strftime("%a %d/%m · %H:%M UTC")
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
        league_name = st.selectbox("🏆 Liga", list(LEAGUES.keys()))
        lc = LEAGUES[league_name]
        days_ahead  = st.slider("📅 Próximos días", 1, 14, 7)
        ev_min_pct  = st.slider("🎯 EV mínimo (%)", 2, 12, 4)
        ev_threshold = ev_min_pct / 100
        st.divider()
        if st.button("🔄 Actualizar datos", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # ── Carga ────────────────────────────────────────────────
    with st.spinner("Cargando datos..."):
        season_df    = fetch_season_matches(FD_KEY, lc["fd"])
        standings_df = fetch_standings(FD_KEY, lc["fd"])
        upcoming     = fetch_upcoming_matches(FD_KEY, lc["fd"], days_ahead)
        odds_list    = fetch_odds(ODDS_KEY, lc["odds"])

    if season_df.empty:
        st.error("No se pudieron cargar datos. Verifica tu FOOTBALL_API_KEY.")
        st.stop()

    ratings, avg_h, avg_a = build_ratings(season_df)

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
        alerts= match_alerts(hctx, actx, md, lc)
        # Form for card display
        hform = team_form(season_df, m["home_id"], 3)
        aform = team_form(season_df, m["away_id"], 3)
        vbets = detect_value_bets(p, bk, m["home_name"], m["away_name"], ev_threshold)
        for vb in vbets:
            vb.update({
                "date":     m["date"],
                "matchday": md,
                "home_ctx": hctx,
                "away_ctx": actx,
                "ctx_alerts": alerts,
                # Extra data for analysis + card precision
                "home_id":   m["home_id"],
                "away_id":   m["away_id"],
                "probs":     p,
                "home_form": hform,
                "away_form": aform,
            })
            all_vb.append(vb)
        match_map[m["id"]] = {"p":p,"bk":bk,"vbets":vbets,"hctx":hctx,"actx":actx,
                               "alerts":alerts,"md":md,"hform":hform,"aform":aform}

    all_vb.sort(key=lambda x: x["ev"], reverse=True)

    # ── Sidebar – Part 2: Summary card (post pre-compute) ─
    render_sidebar_summary(all_vb, lc, ev_min_pct)

    # ── Tabs ─────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs(["🎯 Value Bets","🗓️ Partidos","🔍 Equipo","📋 Clasificación","📈 Tracker"])

    # ─── TAB 1: VALUE BETS ────────────────────────────────────
    with t1:
        st.markdown(f"### 🎯 Value Bets · {league_name}")
        col_l, col_r = st.columns([1,1])
        conf_filter   = col_l.multiselect("Confianza", ["Alta","Media","Baja"], default=["Alta","Media","Baja"])
        market_filter = col_r.multiselect("Mercado",
            ["1 Local","X Empate","2 Visitante","Over 2.5","Under 2.5"],
            default=["1 Local","X Empate","2 Visitante","Over 2.5","Under 2.5"])
        st.markdown("---")

        filtered = [v for v in all_vb if v["conf_label"] in conf_filter and v["label"] in market_filter]

        if not filtered:
            st.info("No se detectaron value bets con estos criterios. Prueba bajando el EV mínimo.")
        else:
            col_badge1, col_badge2, col_badge3 = st.columns(3)
            col_badge1.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_GREEN}'>{sum(1 for v in filtered if v['conf_key']=='high')}</div><div class='stat-card-label'>Alta confianza</div></div>", unsafe_allow_html=True)
            col_badge2.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_STEEL}'>{sum(1 for v in filtered if v['conf_key']=='medium')}</div><div class='stat-card-label'>Media confianza</div></div>", unsafe_allow_html=True)
            col_badge3.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_AMBER}'>{sum(1 for v in filtered if v['conf_key']=='low')}</div><div class='stat-card-label'>Baja confianza</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

            for idx, vb in enumerate(filtered):
                # Animated card
                st.markdown(vb_card_html(vb, idx), unsafe_allow_html=True)
                # Expandable analysis
                with st.expander("📊 ¿Por qué este pick?", expanded=False):
                    analysis = generate_analysis(vb, ratings)
                    st.markdown(analysis)

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
        st.markdown(f"### 🗓️ Próximos {days_ahead} días · {league_name}")
        if not upcoming:
            st.info("No hay partidos en este período.")
        for m in upcoming:
            info  = match_map[m["id"]]
            p,bk  = info["p"], info["bk"]
            hctx,actx,alerts = info["hctx"],info["actx"],info["alerts"]
            dt = datetime.fromisoformat(m["date"].replace("Z","+00:00"))
            suffix = f"  🎯 {len(info['vbets'])}" if info["vbets"] else ""
            suffix += "  ⚠️" if alerts else ""
            with st.expander(
                f"{hctx['emoji']} **{m['home_name']}** vs **{m['away_name']}** {actx['emoji']} "
                f"· {dt.strftime('%a %d/%m %H:%M')} UTC{suffix}", expanded=False):
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
                for vb in info["vbets"]:
                    st.success(f"🎯 **{vb['label']}** @ {vb['bk_odds']} · Modelo {vb['model_p']*100:.1f}% · Implícita {vb['implied']*100:.1f}% · **EV +{vb['ev']*100:.1f}%** · {vb['conf_icon']} {vb['conf_label']}")

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
        sc1, sc2, sc3, sc4 = st.columns(4)
        streak_color = BRAND_GREEN if streak_type == "hit" else ("#ef4444" if streak_type == "miss" else "#94a3b8")
        sc1.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{streak_color}'>{streak_count}</div><div class='stat-card-label'>Racha activa</div></div>", unsafe_allow_html=True)
        sc2.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_GREEN}'>{stats['win_rate']}%</div><div class='stat-card-label'>Win rate · {stats['hits']}H / {stats['misses']}M</div></div>", unsafe_allow_html=True)
        roi_color = BRAND_GREEN if stats['roi'] >= 0 else "#ef4444"
        sc3.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{roi_color}'>{stats['roi']:+.1f}%</div><div class='stat-card-label'>ROI</div></div>", unsafe_allow_html=True)
        pnl_color = BRAND_GREEN if stats['pnl'] >= 0 else "#ef4444"
        sc4.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{pnl_color}'>{stats['pnl']:+.2f}</div><div class='stat-card-label'>P&L · Apostado {stats['staked']:.2f}</div></div>", unsafe_allow_html=True)

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
                odds_val   = fc5.number_input("Cuota", min_value=1.01, max_value=20.0,
                                              value=float(sel_vb["bk_odds"]) if sel_vb else 2.0, step=0.05)
                stake_val  = fc6.number_input("Stake (u.)", min_value=0.0, value=10.0, step=1.0)
                result_val = fc7.selectbox("Resultado", ["Pendiente","Hit ✅","Miss ❌"])
                date_val   = st.date_input("Fecha del partido",
                                           value=datetime.fromisoformat(sel_vb["date"].replace("Z","+00:00")).date() if sel_vb else datetime.utcnow().date())

                submitted = st.form_submit_button("💾 Guardar pick", use_container_width=True)
                if submitted:
                    if not home_val or not away_val:
                        st.error("Completa Local y Visitante.")
                    else:
                        result_map = {"Pendiente":"pending","Hit ✅":"hit","Miss ❌":"miss"}
                        model_p_val = sel_vb["model_p"] if sel_vb else 0.5
                        ev_val      = sel_vb["ev"]      if sel_vb else 0.0
                        add_pick_to_history(
                            home=home_val, away=away_val, league=league_val,
                            market=market_val, odds=odds_val,
                            model_p=model_p_val, ev=ev_val,
                            stake=stake_val,
                            date_match=date_val.isoformat(),
                        )
                        # If result already known, resolve immediately
                        r = result_map[result_val]
                        if r != "pending":
                            new_hist = load_history()
                            if new_hist:
                                resolve_pick(new_hist[-1]["id"], r, stake_val)
                        st.success(f"✅ Pick guardado: {home_val} vs {away_val} · {market_val} @ {odds_val}")
                        st.rerun()

        # ── Pendientes ────────────────────────────────────
        if pending_picks:
            st.markdown(f"#### ⏳ Pendientes de resolución ({len(pending_picks)})")
            for p in sorted(pending_picks, key=lambda x: x.get("date_match",""), reverse=True):
                with st.container():
                    pc1, pc2, pc3, pc4 = st.columns([3,1,1,1])
                    pc1.markdown(
                        f"**{p['home']} vs {p['away']}**  \n"
                        f"<span style='font-size:.75rem;color:#64748b;font-family:IBM Plex Mono'>"
                        f"{p['market']} @ {p['odds']} · EV +{p['ev']}% · Stake {p['stake']}</span>",
                        unsafe_allow_html=True
                    )
                    if pc2.button("✅ Hit",  key=f"hit_{p['id']}"):
                        resolve_pick(p["id"], "hit",  p["stake"]); st.rerun()
                    if pc3.button("❌ Miss", key=f"miss_{p['id']}"):
                        resolve_pick(p["id"], "miss", p["stake"]); st.rerun()
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
            show_cols = ["home","away","league","market","odds","model_p","ev","stake","Resultado","P&L","ROI pick"]
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

    st.markdown("""
    <div class="footer">
      <b>STATIUM</b> · Sports Intelligence. Predict The Edge.<br>
      Poisson Calibrado · Shrinkage Bayesiano · Contexto Competitivo<br>
      Datos: football-data.org · Cuotas: the-odds-api.com · Las probabilidades son estimaciones estadísticas.
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
