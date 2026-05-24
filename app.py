import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Statitum · Value Bets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ────────────────────────────────────────────
BRAND_GREEN  = "#00c896"
BRAND_BLUE   = "#3b82f6"
BRAND_PURPLE = "#6366f1"
BRAND_GOLD   = "#f59e0b"

LOGO_SVG = """
<svg width="52" height="52" viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="lg1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00c896"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>
  <rect width="52" height="52" rx="14" fill="url(#lg1)"/>
  <rect x="9"  y="32" width="7" height="11" rx="2" fill="white" opacity="0.55"/>
  <rect x="19" y="23" width="7" height="20" rx="2" fill="white" opacity="0.75"/>
  <rect x="29" y="15" width="7" height="28" rx="2" fill="white"/>
  <polyline points="12,36 22,27 32,19 39,13" stroke="white" stroke-width="1.5"
            fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.5"/>
  <circle cx="39" cy="13" r="3.5" fill="white"/>
</svg>
"""

import base64
def logo_img(size=52):
    b64 = base64.b64encode(LOGO_SVG.strip().encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" style="border-radius:10px">'

st.markdown(f"""
<style>
  /* ── Reset & Base ── */
  .block-container {{ padding-top: 1.2rem; max-width: 1200px; }}

  /* ── Header ── */
  .stat-header {{
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 8px 0;
  }}
  .stat-logo {{ flex-shrink: 0; }}
  .stat-title {{
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, {BRAND_GREEN}, {BRAND_BLUE});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
  }}
  .stat-subtitle {{ color: #64748b; font-size: 0.88rem; margin-top: 2px; }}

  /* ── Gradient separator ── */
  .grad-line {{
    height: 2px; border-radius: 2px;
    background: linear-gradient(90deg, {BRAND_GREEN}, {BRAND_BLUE}, {BRAND_PURPLE}, transparent);
    margin: 10px 0 18px 0;
  }}

  /* ── Stat cards (top metrics) ── */
  .stat-card {{
    background: white; border-radius: 14px; padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 3px solid {BRAND_GREEN};
    text-align: center;
  }}
  .stat-card-num  {{ font-size: 1.8rem; font-weight: 800; color: #0f172a; }}
  .stat-card-label{{ font-size: 0.75rem; color: #64748b; margin-top: 2px; }}

  /* ── Value Bet Cards ── */
  @keyframes slideInUp {{
    from {{ opacity: 0; transform: translateY(22px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  .vb-card {{
    background: white; border-radius: 16px; padding: 20px 22px;
    margin-bottom: 14px; position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
    transition: transform 0.15s, box-shadow 0.15s;
    animation: slideInUp 0.45s ease-out both;
  }}
  .vb-card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 32px rgba(0,0,0,0.11); }}
  .vb-card-high   {{ border-left: 5px solid #10b981; box-shadow: 0 4px 20px rgba(16,185,129,0.12); }}
  .vb-card-medium {{ border-left: 5px solid {BRAND_GOLD}; box-shadow: 0 4px 20px rgba(245,158,11,0.10); }}
  .vb-card-low    {{ border-left: 5px solid #f97316; box-shadow: 0 4px 20px rgba(249,115,22,0.08); }}

  .vb-match   {{ font-size: 1.05rem; font-weight: 700; color: #0f172a; margin: 0; }}
  .vb-meta    {{ font-size: 0.78rem; color: #64748b; margin-top: 3px; }}
  .vb-ctx-row {{ display: flex; gap: 6px; margin: 8px 0; flex-wrap: wrap; }}

  /* ── EV Badge ── */
  .ev-pill {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 1.25rem; font-weight: 800;
    padding: 6px 16px; border-radius: 30px;
  }}
  .ev-high   {{ background: linear-gradient(135deg,#d1fae5,#a7f3d0); color: #065f46; }}
  .ev-medium {{ background: linear-gradient(135deg,#fef3c7,#fde68a); color: #78350f; }}
  .ev-low    {{ background: linear-gradient(135deg,#ffedd5,#fed7aa); color: #7c2d12; }}

  .conf-tag {{
    font-size: 0.75rem; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; display: inline-block;
  }}
  .conf-high   {{ background: #d1fae5; color: #065f46; }}
  .conf-medium {{ background: #fef3c7; color: #78350f; }}
  .conf-low    {{ background: #ffedd5; color: #7c2d12; }}

  .vb-details {{
    font-size: 0.82rem; color: #475569; margin-top: 10px;
    display: flex; flex-wrap: wrap; gap: 12px;
  }}
  .vb-detail-item {{ display: flex; flex-direction: column; }}
  .vb-detail-label {{ font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .5px; }}
  .vb-detail-val   {{ font-weight: 700; color: #0f172a; font-size: 0.92rem; }}
  .vb-detail-val.green {{ color: #059669; }}
  .vb-detail-val.blue  {{ color: #3b82f6; }}

  /* ── Context badges ── */
  .ctx-badge  {{ display:inline-block; font-size:.72rem; font-weight:600; padding:2px 9px; border-radius:12px; }}
  .ctx-title      {{ background:#fef9c3; color:#854d0e; border:1px solid #fde047; }}
  .ctx-champion   {{ background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }}
  .ctx-champ-won  {{ background:#f0fdf4; color:#166534; border:1px solid #86efac; }}
  .ctx-europa     {{ background:#f0fdf4; color:#166534; border:1px solid #86efac; }}
  .ctx-mid        {{ background:#f8fafc; color:#64748b; border:1px solid #e2e8f0; }}
  .ctx-dead       {{ background:#f8fafc; color:#94a3b8; border:1px solid #e2e8f0; font-style:italic; }}
  .ctx-nearrel    {{ background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; }}
  .ctx-relegation {{ background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }}

  /* ── Context alert ── */
  .ctx-alert {{
    background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px;
    padding: 7px 12px; font-size: .80rem; color: #92400e; margin-top: 8px;
  }}

  /* ── Compact prob bar (inside card) ── */
  .card-prob-bar {{
    display:flex; height:22px; border-radius:6px; overflow:hidden; gap:1px;
    margin: 10px 0 4px;
  }}
  .cpb-home {{ background:linear-gradient(135deg,#10b981,#34d399); display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:700; min-width:28px; }}
  .cpb-draw {{ background:linear-gradient(135deg,#94a3b8,#cbd5e1); display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:700; min-width:28px; }}
  .cpb-away {{ background:linear-gradient(135deg,#3b82f6,#60a5fa); display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:700; min-width:28px; }}

  /* ── Full probability bars (Tab 2) ── */
  .prob-bar-wrap {{ margin: 14px 0 4px; }}
  .prob-bar-label {{ font-size: .70rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 5px; }}
  .prob-bar-1x2 {{ display:flex; height:28px; border-radius:8px; overflow:hidden; gap:1px; }}
  .pb-home {{ background: linear-gradient(135deg,#10b981,#34d399); display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:700; min-width:30px; }}
  .pb-draw {{ background: linear-gradient(135deg,#94a3b8,#cbd5e1); display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:700; min-width:30px; }}
  .pb-away {{ background: linear-gradient(135deg,#3b82f6,#60a5fa); display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:700; min-width:30px; }}

  .prob-pills {{ display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }}
  .prob-pill {{
    display:flex; align-items:center; gap:5px; padding:4px 10px;
    background:#f8fafc; border-radius:20px; font-size:.78rem;
    border:1px solid #e2e8f0;
  }}
  .prob-pill-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
  .pp-green  {{ background:#10b981; }}
  .pp-blue   {{ background:#3b82f6; }}
  .pp-orange {{ background:#f97316; }}
  .pp-purple {{ background:#8b5cf6; }}
  .prob-pill-val {{ font-weight:700; color:#0f172a; }}

  /* ── Form badges ── */
  .form-badge {{ display:inline-block; width:26px; height:26px; border-radius:6px; text-align:center; line-height:26px; font-size:12px; font-weight:700; color:white; margin:1px; }}
  .fb-w {{ background:#10b981; }}
  .fb-d {{ background:#94a3b8; }}
  .fb-l {{ background:#ef4444; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{ background: white !important; }}

  /* ── Tabs ── */
  [data-testid="stTabs"] button {{ font-weight:600; }}

  /* ── Override Streamlit metric ── */
  div[data-testid="stMetricValue"] {{ font-size:1.5rem !important; font-weight:800 !important; }}

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

    # Compact 1X2 probability bar
    prob_html = ""
    p = vb.get("probs")
    if p:
        h_pct = p["home_win"]*100
        d_pct = p["draw"]*100
        a_pct = p["away_win"]*100
        prob_html = f"""
        <div style="margin:12px 0 6px">
          <div style="font-size:.66rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px">
            Prob. 1·X·2 &nbsp;·&nbsp; xG: <b>{p['lam_h']}</b> – <b>{p['lam_a']}</b> &nbsp;·&nbsp; Over 2.5: <b>{p['over25']*100:.0f}%</b> &nbsp;·&nbsp; BTTS: <b>{p['btts']*100:.0f}%</b>
          </div>
          <div class="card-prob-bar">
            <div class="cpb-home" style="width:{h_pct:.1f}%">{h_pct:.0f}%</div>
            <div class="cpb-draw" style="width:{d_pct:.1f}%">{d_pct:.0f}%</div>
            <div class="cpb-away" style="width:{a_pct:.1f}%">{a_pct:.0f}%</div>
          </div>
        </div>"""

    # Mini form row
    form_h = vb.get("home_form", [])
    form_a = vb.get("away_form", [])
    form_html = ""
    if form_h or form_a:
        fh_badges = render_form_mini(form_h)
        fa_badges = render_form_mini(form_a)
        form_html = f"""
        <div style="display:flex;gap:18px;margin-top:8px;align-items:center;flex-wrap:wrap">
          <div style="display:flex;align-items:center;gap:5px">
            <span style="font-size:.68rem;color:#94a3b8;white-space:nowrap">🏠 Forma</span>{fh_badges}
          </div>
          <div style="display:flex;align-items:center;gap:5px">
            <span style="font-size:.68rem;color:#94a3b8;white-space:nowrap">✈️ Forma</span>{fa_badges}
          </div>
        </div>"""

    return f"""
    <div class="vb-card vb-card-{vb['conf_key']}" style="animation-delay:{delay:.2f}s">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
        <div>
          <p class="vb-match">{vb['home']} <span style="color:#94a3b8;font-weight:400">vs</span> {vb['away']}</p>
          <p class="vb-meta">📅 {date_str} &nbsp;·&nbsp; Jornada {vb.get('matchday','?')} &nbsp;·&nbsp; Mercado: <b>{vb['label']}</b></p>
          <div class="vb-ctx-row">{ctx_badge_html(hctx)} <span style="color:#cbd5e1;font-size:.75rem;align-self:center">vs</span> {ctx_badge_html(actx)}</div>
        </div>
        <div style="text-align:right">
          <div class="{vb['ev_css']} ev-pill">📈 EV +{ev_pct:.1f}%</div>
          <div style="margin-top:6px"><span class="conf-tag {vb['conf_css']}">{vb['conf_icon']} Confianza {vb['conf_label']}</span></div>
        </div>
      </div>
      {prob_html}
      <div class="vb-details">
        <div class="vb-detail-item"><span class="vb-detail-label">Cuota disponible</span><span class="vb-detail-val green">📌 {vb['bk_odds']}</span></div>
        <div class="vb-detail-item"><span class="vb-detail-label">Cuota justa</span><span class="vb-detail-val">{round(1/vb['model_p'],2)}</span></div>
        <div class="vb-detail-item"><span class="vb-detail-label">P. modelo</span><span class="vb-detail-val">{vb['model_p']*100:.1f}%</span></div>
        <div class="vb-detail-item"><span class="vb-detail-label">P. implícita</span><span class="vb-detail-val">{vb['implied']*100:.1f}%</span></div>
        <div class="vb-detail-item"><span class="vb-detail-label">Edge</span><span class="vb-detail-val green">+{edge_pp:.1f}pp</span></div>
        <div class="vb-detail-item"><span class="vb-detail-label">xG Local</span><span class="vb-detail-val blue">{vb['probs']['lam_h'] if vb.get('probs') else '—'}</span></div>
        <div class="vb-detail-item"><span class="vb-detail-label">xG Visit.</span><span class="vb-detail-val blue">{vb['probs']['lam_a'] if vb.get('probs') else '—'}</span></div>
      </div>
      {form_html}
      {alert_html}
    </div>"""

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
            <div style="background:linear-gradient(90deg,#00c896,#3b82f6);width:{pct}%;height:100%;border-radius:4px"></div>
          </div>
        </div>"""

    decay_pct    = int(DECAY_RATE * 1000)        # 0.010 → 10 (out of ~20)
    shrink_pct   = int(SHRINKAGE_K / 20 * 100)  # 10 → 50%
    edge_pct_bar = int(MAX_EDGE * 100 / 25 * 100) # 17/25 → 68%
    ev_bar_pct   = int(ev_min_pct / 15 * 100)   # scaled to 15% max

    with st.sidebar:
        # ── Summary card ──
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:16px;padding:18px;margin:4px 0 16px">
          <div style="color:#64748b;font-size:.68rem;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Picks detectados</div>
          <div style="font-size:2.4rem;font-weight:800;color:white;line-height:1">{n_total}</div>
          <div style="color:#475569;font-size:.72rem;margin-bottom:14px">value bets · EV medio +{avg_ev}%</div>
          <div style="display:flex;gap:8px">
            <div style="flex:1;background:rgba(16,185,129,0.18);border-radius:10px;padding:8px 6px;text-align:center;border:1px solid rgba(16,185,129,0.25)">
              <div style="color:#34d399;font-size:1.3rem;font-weight:800">{n_high}</div>
              <div style="color:#6ee7b7;font-size:.63rem;margin-top:1px">🟢 Alta</div>
            </div>
            <div style="flex:1;background:rgba(245,158,11,0.18);border-radius:10px;padding:8px 6px;text-align:center;border:1px solid rgba(245,158,11,0.25)">
              <div style="color:#fcd34d;font-size:1.3rem;font-weight:800">{n_medium}</div>
              <div style="color:#fde68a;font-size:.63rem;margin-top:1px">🟡 Media</div>
            </div>
            <div style="flex:1;background:rgba(249,115,22,0.18);border-radius:10px;padding:8px 6px;text-align:center;border:1px solid rgba(249,115,22,0.25)">
              <div style="color:#fb923c;font-size:1.3rem;font-weight:800">{n_low}</div>
              <div style="color:#fed7aa;font-size:.63rem;margin-top:1px">🟠 Baja</div>
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
        <div class="stat-title">STATITUM</div>
        <div class="stat-subtitle">Análisis estadístico deportivo · Detección de value bets con EV real</div>
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
          <span style="font-weight:800;font-size:1.1rem;background:linear-gradient(135deg,#00c896,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent">STATITUM</span>
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
    t1, t2, t3, t4 = st.tabs(["🎯 Value Bets","🗓️ Partidos","🔍 Equipo","📋 Clasificación"])

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
            col_badge1.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:#10b981'>{sum(1 for v in filtered if v['conf_key']=='high')}</div><div class='stat-card-label'>🟢 Alta confianza</div></div>", unsafe_allow_html=True)
            col_badge2.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:{BRAND_GOLD}'>{sum(1 for v in filtered if v['conf_key']=='medium')}</div><div class='stat-card-label'>🟡 Media confianza</div></div>", unsafe_allow_html=True)
            col_badge3.markdown(f"<div class='stat-card'><div class='stat-card-num' style='color:#f97316'>{sum(1 for v in filtered if v['conf_key']=='low')}</div><div class='stat-card-label'>🟠 Baja confianza</div></div>", unsafe_allow_html=True)

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

    st.markdown("""
    <div class="footer">
      <b>STATITUM</b> · Poisson Calibrado · Shrinkage Bayesiano · Contexto Competitivo<br>
      Datos: football-data.org · Cuotas: the-odds-api.com · Las probabilidades son estimaciones estadísticas.
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
