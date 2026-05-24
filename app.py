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

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }

  .vb-high   { background:linear-gradient(135deg,#1a472a,#2d6a4f); border-radius:12px; padding:18px; color:white; margin-bottom:12px; border-left:5px solid #52b788; }
  .vb-medium { background:linear-gradient(135deg,#3d3500,#665c00); border-radius:12px; padding:18px; color:white; margin-bottom:12px; border-left:5px solid #ffd60a; }
  .vb-low    { background:linear-gradient(135deg,#2c1810,#5a3020); border-radius:12px; padding:18px; color:white; margin-bottom:12px; border-left:5px solid #e07a5f; }

  .ev-badge          { font-weight:700; padding:3px 11px; border-radius:20px; font-size:.95rem; }
  .badge-high        { background:#52b788; color:#fff; }
  .badge-medium      { background:#ffd60a; color:#333; }
  .badge-low         { background:#e07a5f; color:#fff; }
  .conf-label        { font-size:.78rem; opacity:.85; margin-left:6px; }

  .ctx-badge         { display:inline-block; font-size:.75rem; font-weight:600; padding:2px 8px; border-radius:12px; margin:2px; }
  .ctx-title         { background:#ffe066; color:#3d2c00; }
  .ctx-champions     { background:#b3d1ff; color:#00226e; }
  .ctx-europa        { background:#c3f0ca; color:#0a3d1a; }
  .ctx-mid           { background:#404040; color:#ccc; }
  .ctx-dead          { background:#555; color:#bbb; font-style:italic; }
  .ctx-nearrel       { background:#ffb347; color:#3d1a00; }
  .ctx-relegation    { background:#ff6b6b; color:#fff; }

  .ctx-alert         { background:#7a2200; border:1px solid #ff6b35; border-radius:8px; padding:8px 12px; font-size:.83rem; margin-top:8px; color:#ffd5b8; }

  .prob-row  { display:flex; gap:8px; margin-top:8px; }
  .prob-box  { flex:1; text-align:center; border-radius:8px; padding:10px 4px; font-weight:600; font-size:.88rem; }
  .home-box  { background:#d8f3dc; color:#1b4332; }
  .draw-box  { background:#fff3cd; color:#664d03; }
  .away-box  { background:#d0e8f7; color:#0a3d62; }
  .over-box  { background:#fde8d8; color:#7f3f00; }
  .under-box { background:#e8d8fd; color:#3f007f; }
  .btts-box  { background:#d8eafd; color:#003f7f; }

  div[data-testid="stMetricValue"] { font-size:1.55rem !important; }
  .footer { text-align:center; color:#888; font-size:.76rem; margin-top:2rem; }
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

MIN_ODDS      = 1.35
MAX_ODDS      = 7.00
MAX_EDGE      = 0.17
SHRINKAGE_K   = 10
DECAY_RATE    = 0.010
LATE_SEASON_MD = 5   # Últimas N jornadas = "temporada tardía"

# ═══════════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════════

def _fd_get(api_key, endpoint, params=None):
    try:
        r = requests.get(f"{FD_BASE}{endpoint}",
                         headers={"X-Auth-Token": api_key},
                         params=params, timeout=12)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def _odds_get(api_key, endpoint, params=None):
    try:
        r = requests.get(f"{ODDS_BASE}{endpoint}",
                         params={"apiKey": api_key, **(params or {})},
                         timeout=12)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_season_matches(fd_key, fd_code):
    data = _fd_get(fd_key, f"/competitions/{fd_code}/matches", {"status":"FINISHED"})
    if not data:
        return pd.DataFrame()
    rows = []
    for m in data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        if ft.get("home") is not None:
            rows.append({
                "date":       m["utcDate"],
                "home_id":    m["homeTeam"]["id"],
                "home_name":  m["homeTeam"]["name"],
                "away_id":    m["awayTeam"]["id"],
                "away_name":  m["awayTeam"]["name"],
                "home_goals": ft["home"],
                "away_goals": ft["away"],
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_upcoming_matches(fd_key, fd_code, days=7):
    today  = datetime.utcnow().strftime("%Y-%m-%d")
    future = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
    data = _fd_get(fd_key, f"/competitions/{fd_code}/matches",
                   {"status":"SCHEDULED","dateFrom":today,"dateTo":future})
    if not data:
        return []
    return [{
        "id":        m["id"],
        "date":      m["utcDate"],
        "home_id":   m["homeTeam"]["id"],
        "home_name": m["homeTeam"]["name"],
        "away_id":   m["awayTeam"]["id"],
        "away_name": m["awayTeam"]["name"],
        "matchday":  m.get("matchday") or 0,
    } for m in data.get("matches", [])]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_standings(fd_key, fd_code):
    """Tabla de posiciones actual de la liga."""
    data = _fd_get(fd_key, f"/competitions/{fd_code}/standings")
    if not data:
        return pd.DataFrame()
    for s in data.get("standings", []):
        if s.get("type") == "TOTAL":
            rows = []
            for e in s.get("table", []):
                rows.append({
                    "position":  e["position"],
                    "team_id":   e["team"]["id"],
                    "team_name": e["team"]["name"],
                    "played":    e["playedGames"],
                    "points":    e["points"],
                    "gf":        e["goalsFor"],
                    "ga":        e["goalsAgainst"],
                    "gd":        e["goalDifference"],
                })
            return pd.DataFrame(rows)
    return pd.DataFrame()

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_odds(odds_key, sport_key):
    data = _odds_get(odds_key, f"/sports/{sport_key}/odds/",
                     {"regions":"eu","markets":"h2h,totals","oddsFormat":"decimal"})
    return data if isinstance(data, list) else []

# ═══════════════════════════════════════════════════════════
# CONTEXTO COMPETITIVO
# ═══════════════════════════════════════════════════════════

def get_team_context(team_id, standings_df, league_cfg, matchday):
    """
    Determina el contexto competitivo de un equipo:
    título / Champions / Europa / media tabla / sin motivación / descenso.
    También detecta si es temporada tardía (últimas N jornadas).
    """
    if standings_df.empty:
        return {"label":"Sin datos","emoji":"❓","css":"ctx-mid","alert":False,"dead":False}

    row = standings_df[standings_df["team_id"] == team_id]
    if row.empty:
        return {"label":"Sin datos","emoji":"❓","css":"ctx-mid","alert":False,"dead":False}

    row        = row.iloc[0]
    pos        = int(row["position"])
    pts        = int(row["points"])
    played     = int(row["played"])
    total_t    = league_cfg["teams"]
    total_g    = league_cfg["games"]
    remaining  = max(0, total_g - played)
    cl_spots   = league_cfg["cl"]
    euro_spots = league_cfg["euro"]
    rel_spots  = league_cfg["rel"]
    safe_pos   = total_t - rel_spots  # última posición segura

    # Puntos de referencia
    def pts_at(position):
        r = standings_df[standings_df["position"] == position]
        return int(r["points"].iloc[0]) if not r.empty else 0

    pts_rel_border = pts_at(safe_pos + 1)  # 1ª posición de descenso
    pts_euro_border = pts_at(euro_spots)
    pts_cl_border   = pts_at(cl_spots)

    late = matchday >= (total_g - LATE_SEASON_MD + 1)

    # ── Título ──
    if pos <= 2:
        return {"label":"Pelea título","emoji":"🏆","css":"ctx-title","alert":True,"dead":False}

    # ── Champions ──
    if pos <= cl_spots:
        return {"label":"Zona Champions","emoji":"⭐","css":"ctx-champions","alert":True,"dead":False}

    # ── Persigue Champions ──
    if pos == cl_spots + 1 and pts >= pts_cl_border - remaining * 3:
        return {"label":"Persigue Champions","emoji":"⭐","css":"ctx-champions","alert":True,"dead":False}

    # ── Europa ──
    if pos <= euro_spots:
        return {"label":"Zona Europa","emoji":"🌍","css":"ctx-europa","alert":True,"dead":False}

    # ── Descenso ──
    if pos > safe_pos:
        return {"label":"Zona descenso","emoji":"🔴","css":"ctx-relegation","alert":True,"dead":False}

    # ── Pelea descenso (a 3 pts o menos del borde) ──
    if pts - pts_rel_border <= 3:
        return {"label":"Pelea descenso","emoji":"🟠","css":"ctx-nearrel","alert":True,"dead":False}

    # ── Sin motivación extra (zona muerta) ──
    if late and pts > pts_rel_border + remaining * 3 and pos > euro_spots:
        return {"label":"Sin motivación extra","emoji":"😴","css":"ctx-dead","alert":True,"dead":True}

    return {"label":"Zona media","emoji":"➖","css":"ctx-mid","alert":False,"dead":False}


def match_context_alert(home_ctx, away_ctx, matchday, league_cfg):
    """
    Genera un mensaje de alerta cuando hay incertidumbre motivacional en el partido.
    """
    alerts = []
    total_g = league_cfg["games"]
    remaining_approx = total_g - matchday

    if remaining_approx <= 3:
        alerts.append(f"⚠️ Jornada {matchday}/{total_g} — últimas jornadas de temporada. "
                      "El modelo estadístico NO considera motivaciones ni rotaciones.")

    if home_ctx["dead"] and away_ctx["dead"]:
        alerts.append("😴 Ambos equipos sin motivación real — resultado estadístico muy incierto.")
    elif home_ctx["dead"]:
        alerts.append(f"😴 {home_ctx['label']} para el local — posibles rotaciones o falta de intensidad.")
    elif away_ctx["dead"]:
        alerts.append(f"😴 {away_ctx['label']} para el visitante — posibles rotaciones o falta de intensidad.")

    if home_ctx["css"] == "ctx-relegation" and away_ctx["dead"]:
        alerts.append("🔴 Descenso vs sin motivación — el equipo local tiene mucho más que perder. Alta varianza.")
    elif away_ctx["css"] == "ctx-relegation" and home_ctx["dead"]:
        alerts.append("🔴 Sin motivación vs descenso — el visitante llegará con hambre. Alta varianza.")

    return alerts

# ═══════════════════════════════════════════════════════════
# MODELO POISSON CALIBRADO
# ═══════════════════════════════════════════════════════════

def build_ratings(df):
    if df.empty or len(df) < 20:
        return {}, 1.35, 1.10

    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["days_ago"] = (now - df["date_parsed"]).dt.days.clip(lower=0)
    df["w"] = np.exp(-DECAY_RATE * df["days_ago"])

    avg_h = float(np.average(df["home_goals"].values, weights=df["w"].values))
    avg_a = float(np.average(df["away_goals"].values, weights=df["w"].values))
    if avg_h == 0 or avg_a == 0:
        return {}, 1.35, 1.10

    mean_w = df["w"].mean()

    def wavg(vals, wts, fallback=1.0):
        s = wts.sum()
        return float(np.average(vals, weights=wts)) if (len(vals) >= 1 and s > 0) else fallback

    def shrink(raw, n_eff):
        return (n_eff * raw + SHRINKAGE_K * 1.0) / (n_eff + SHRINKAGE_K)

    ratings = {}
    for tid in set(df["home_id"]) | set(df["away_id"]):
        hm = df[df["home_id"] == tid]
        am = df[df["away_id"] == tid]
        nh, na = len(hm), len(am)
        if nh + na < 4:
            continue
        n_eff_h = hm["w"].sum() / mean_w if nh >= 1 else 0.0
        n_eff_a = am["w"].sum() / mean_w if na >= 1 else 0.0

        att_h = shrink(wavg(hm["home_goals"].values, hm["w"].values) / avg_h if nh >= 2 else 1.0, n_eff_h)
        att_a = shrink(wavg(am["away_goals"].values, am["w"].values) / avg_a if na >= 2 else 1.0, n_eff_a)
        def_h = shrink(wavg(hm["away_goals"].values, hm["w"].values) / avg_a if nh >= 2 else 1.0, n_eff_h)
        def_a = shrink(wavg(am["home_goals"].values, am["w"].values) / avg_h if na >= 2 else 1.0, n_eff_a)

        gs_avg = (hm["home_goals"].sum() + am["away_goals"].sum()) / (nh + na)
        gc_avg = (hm["away_goals"].sum() + am["home_goals"].sum()) / (nh + na)

        ratings[tid] = {
            "att_h":round(att_h,3),"att_a":round(att_a,3),
            "def_h":round(def_h,3),"def_a":round(def_a,3),
            "gs_avg":round(gs_avg,2),"gc_avg":round(gc_avg,2),
            "n":nh+na,"n_eff":round(n_eff_h+n_eff_a,1),
        }
    return ratings, avg_h, avg_a


def match_probs(home_id, away_id, ratings, avg_h, avg_a):
    if home_id not in ratings or away_id not in ratings:
        return None
    hr, ar = ratings[home_id], ratings[away_id]
    lam_h = float(np.clip(hr["att_h"] * ar["def_a"] * avg_h, 0.40, 4.5))
    lam_a = float(np.clip(ar["att_a"] * hr["def_h"] * avg_a, 0.40, 4.5))

    G  = 8
    ph = [poisson.pmf(i, lam_h) for i in range(G)]
    pa = [poisson.pmf(i, lam_a) for i in range(G)]
    M  = np.outer(ph, pa)

    home_win = float(np.sum(np.tril(M, -1)))
    draw     = float(np.sum(np.diag(M)))
    away_win = float(np.sum(np.triu(M, 1)))
    over25   = float(sum(M[i][j] for i in range(G) for j in range(G) if i+j > 2))
    under25  = 1.0 - over25
    btts     = float((1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a)))

    def fair(p): return round(1/p, 2) if p > 0.01 else 99.0

    return {
        "lam_h":round(lam_h,2),"lam_a":round(lam_a,2),
        "home_win":round(home_win,4),"draw":round(draw,4),"away_win":round(away_win,4),
        "over25":round(over25,4),"under25":round(under25,4),"btts":round(btts,4),
        "fair_1":fair(home_win),"fair_x":fair(draw),"fair_2":fair(away_win),
        "fair_o25":fair(over25),"fair_u25":fair(under25),"fair_btts":fair(btts),
    }


def team_form(df, team_id, n=6):
    hm = df[df["home_id"]==team_id][["date","home_goals","away_goals"]].rename(columns={"home_goals":"gf","away_goals":"gc"})
    hm["v"] = "H"
    am = df[df["away_id"]==team_id][["date","home_goals","away_goals"]].rename(columns={"away_goals":"gf","home_goals":"gc"})
    am["v"] = "A"
    combined = pd.concat([hm, am]).sort_values("date", ascending=False).head(n)
    result = []
    for _, r in combined.iterrows():
        if   r.gf > r.gc: result.append(("W","#2d6a4f",r.gf,r.gc,r.v))
        elif r.gf == r.gc: result.append(("D","#b5940a",r.gf,r.gc,r.v))
        else:              result.append(("L","#a4262c",r.gf,r.gc,r.v))
    return result

# ═══════════════════════════════════════════════════════════
# MOTOR DE VALUE BETS
# ═══════════════════════════════════════════════════════════

def _sim(a, b):
    a = a.lower().replace(" fc","").replace(" cf","").replace(" afc","").replace(" sc","").strip()
    b = b.lower().replace(" fc","").replace(" cf","").replace(" afc","").replace(" sc","").strip()
    return SequenceMatcher(None, a, b).ratio()

def find_odds_match(fd_home, fd_away, fd_date_str, odds_list):
    fd_date = datetime.fromisoformat(fd_date_str.replace("Z","+00:00")).date()
    best, best_score = None, 0
    for om in odds_list:
        try:
            om_date = datetime.fromisoformat(om["commence_time"].replace("Z","+00:00")).date()
        except Exception:
            continue
        if abs((om_date - fd_date).days) > 1:
            continue
        score = _sim(fd_home, om["home_team"]) + _sim(fd_away, om["away_team"])
        if score > best_score and score > 1.25:
            best_score, best = score, om
    return best

def best_odds_for(om):
    best = {"h2h_1":0,"h2h_x":0,"h2h_2":0,"o25":0,"u25":0}
    if not om:
        return best
    ht, at = om.get("home_team",""), om.get("away_team","")
    for bk in om.get("bookmakers",[]):
        for mkt in bk.get("markets",[]):
            if mkt["key"] == "h2h":
                for oc in mkt.get("outcomes",[]):
                    p, n = float(oc.get("price",0)), oc.get("name","")
                    if   _sim(n, ht) > 0.7: best["h2h_1"] = max(best["h2h_1"], p)
                    elif _sim(n, at) > 0.7: best["h2h_2"] = max(best["h2h_2"], p)
                    elif "draw" in n.lower(): best["h2h_x"] = max(best["h2h_x"], p)
            elif mkt["key"] == "totals":
                for oc in mkt.get("outcomes",[]):
                    pt, pr = float(oc.get("point",0)), float(oc.get("price",0))
                    if abs(pt-2.5) < 0.01:
                        if oc.get("name","").lower()=="over":  best["o25"] = max(best["o25"], pr)
                        elif oc.get("name","").lower()=="under": best["u25"] = max(best["u25"], pr)
    return best

def confidence_label(edge):
    if   edge <= 0.08: return "Alta",  "high",   "🟢"
    elif edge <= 0.13: return "Media", "medium", "🟡"
    else:              return "Baja",  "low",    "🟠"

def detect_value_bets(probs, bk_odds, home_name, away_name, ev_threshold):
    if not probs:
        return []
    checks = [
        ("1 Local",     probs["home_win"], bk_odds["h2h_1"]),
        ("X Empate",    probs["draw"],     bk_odds["h2h_x"]),
        ("2 Visitante", probs["away_win"], bk_odds["h2h_2"]),
        ("Over 2.5",    probs["over25"],   bk_odds["o25"]),
        ("Under 2.5",   probs["under25"],  bk_odds["u25"]),
    ]
    found = []
    for label, model_p, bk_odd in checks:
        if bk_odd < MIN_ODDS or bk_odd > MAX_ODDS:
            continue
        ev = model_p * bk_odd - 1
        if ev < ev_threshold:
            continue
        implied = 1 / bk_odd
        edge    = model_p - implied
        if edge > MAX_EDGE:
            continue
        cl, ck, ci = confidence_label(edge)
        found.append({
            "label":label, "model_p":round(model_p,4),
            "implied":round(implied,4), "edge":round(edge,4),
            "bk_odds":bk_odd, "ev":round(ev,4),
            "conf_label":cl, "conf_key":ck, "conf_icon":ci,
            "home":home_name, "away":away_name,
        })
    return found

# ═══════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════

def badges(form_list):
    html = ""
    for res, color, gf, gc, v in form_list:
        html += (f'<span style="background:{color};color:white;padding:3px 9px;'
                 f'border-radius:5px;margin:2px;font-weight:700;font-size:.82rem"'
                 f' title="{gf:.0f}-{gc:.0f} ({v})">{res}</span>')
    return html

def ctx_badge(ctx):
    return (f'<span class="ctx-badge {ctx["css"]}">'
            f'{ctx["emoji"]} {ctx["label"]}</span>')

def prob_html(label, value, css_class):
    return (f'<div class="prob-box {css_class}">'
            f'{label}<br><b>{value*100:.1f}%</b></div>')

def match_probs_html(p, home_name, away_name):
    r1 = (prob_html(f"🏠 {home_name[:13]}", p["home_win"], "home-box") +
          prob_html("🤝 Empate",             p["draw"],     "draw-box") +
          prob_html(f"✈️ {away_name[:13]}",  p["away_win"], "away-box"))
    r2 = (prob_html("Over 2.5",  p["over25"],  "over-box") +
          prob_html("Under 2.5", p["under25"], "under-box") +
          prob_html("BTTS",      p["btts"],    "btts-box"))
    return (f'<div class="prob-row">{r1}</div>'
            f'<div class="prob-row" style="margin-top:6px">{r2}</div>')

# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════

def main():
    st.markdown("## 📊 Statitum")
    st.caption("Análisis estadístico deportivo · Detección de value bets con EV real")
    st.divider()

    try:
        FD_KEY   = st.secrets["FOOTBALL_API_KEY"]
        ODDS_KEY = st.secrets["ODDS_API_KEY"]
    except Exception:
        st.error("⚠️ Faltan las API Keys en **Settings → Secrets**.")
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        league_name = st.selectbox("Liga", list(LEAGUES.keys()))
        league_cfg  = LEAGUES[league_name]
        fd_code     = league_cfg["fd"]
        odds_sport  = league_cfg["odds"]
        days_ahead  = st.slider("Próximos días", 1, 14, 7)
        ev_min_pct  = st.slider("EV mínimo (%)", 2, 12, 4)
        ev_threshold = ev_min_pct / 100

        st.divider()
        st.markdown("### 🧠 Modelo")
        st.success(f"Shrinkage bayesiano K={SHRINKAGE_K}")
        st.info(f"Ponderación temporal λ={DECAY_RATE}")
        st.warning(f"Filtro credibilidad ≤{int(MAX_EDGE*100)}pp")

        st.divider()
        st.markdown("### 🗺️ Contexto competitivo")
        st.markdown(
            "🏆 Título &nbsp; ⭐ Champions &nbsp; 🌍 Europa\n\n"
            "➖ Media tabla &nbsp; 😴 Sin motivación\n\n"
            "🟠 Pelea descenso &nbsp; 🔴 Descenso"
        )
        st.divider()
        if st.button("🔄 Limpiar caché"):
            st.cache_data.clear()
            st.rerun()

    # ── Carga de datos ───────────────────────────────────────
    with st.spinner("📡 Cargando datos de temporada y clasificación..."):
        season_df   = fetch_season_matches(FD_KEY, fd_code)
        standings_df = fetch_standings(FD_KEY, fd_code)

    if season_df.empty:
        st.error("No se pudieron cargar datos. Verifica tu FOOTBALL_API_KEY.")
        st.stop()

    ratings, avg_h, avg_a = build_ratings(season_df)

    with st.spinner("🗓️ Cargando próximos partidos..."):
        upcoming = fetch_upcoming_matches(FD_KEY, fd_code, days_ahead)
    with st.spinner("💰 Cargando cuotas..."):
        odds_list = fetch_odds(ODDS_KEY, odds_sport)

    # Stats bar
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Partidos históricos",  len(season_df))
    c2.metric("Equipos con rating",   len(ratings))
    c3.metric("Próximos partidos",     len(upcoming))
    c4.metric("Mercados con cuotas",   len(odds_list))
    c5.metric("Equipos en tabla",      len(standings_df))
    st.divider()

    # ── Pre-computar todo ────────────────────────────────────
    match_map = {}
    all_vb    = []
    for m in upcoming:
        p      = match_probs(m["home_id"], m["away_id"], ratings, avg_h, avg_a)
        om     = find_odds_match(m["home_name"], m["away_name"], m["date"], odds_list)
        bk     = best_odds_for(om)
        vbets  = detect_value_bets(p, bk, m["home_name"], m["away_name"], ev_threshold)
        md     = int(m["matchday"]) if m["matchday"] else 0

        home_ctx = get_team_context(m["home_id"], standings_df, league_cfg, md)
        away_ctx = get_team_context(m["away_id"], standings_df, league_cfg, md)
        alerts   = match_context_alert(home_ctx, away_ctx, md, league_cfg)

        for vb in vbets:
            vb["date"]      = m["date"]
            vb["matchday"]  = md
            vb["home_ctx"]  = home_ctx
            vb["away_ctx"]  = away_ctx
            vb["ctx_alerts"] = alerts
            all_vb.append(vb)

        match_map[m["id"]] = {
            "p":p,"bk":bk,"vbets":vbets,
            "home_ctx":home_ctx,"away_ctx":away_ctx,"alerts":alerts,"md":md
        }

    all_vb.sort(key=lambda x: x["ev"], reverse=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab_vb, tab_all, tab_team, tab_table = st.tabs([
        "🎯 Value Bets", "🗓️ Todos los Partidos",
        "🔍 Análisis de Equipo", "📋 Clasificación"
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1 · VALUE BETS
    # ─────────────────────────────────────────────────────────
    with tab_vb:
        st.markdown(f"### Value Bets detectados · {league_name}")
        st.caption(f"EV mínimo: **{ev_min_pct}%** · Cuotas: {MIN_ODDS}–{MAX_ODDS} · Edge máx.: {int(MAX_EDGE*100)}pp")

        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown("🟢 **Alta confianza** — edge < 8pp")
        lc2.markdown("🟡 **Media confianza** — edge 8–13pp")
        lc3.markdown("🟠 **Baja confianza** — edge 13–17pp")
        st.divider()

        if not all_vb:
            st.info("No se detectaron value bets. Prueba bajando el EV mínimo o ampliando los días.")
        else:
            fc1, fc2 = st.columns(2)
            conf_filter   = fc1.multiselect("Confianza", ["Alta","Media","Baja"], default=["Alta","Media","Baja"])
            market_filter = fc2.multiselect("Mercado",
                ["1 Local","X Empate","2 Visitante","Over 2.5","Under 2.5"],
                default=["1 Local","X Empate","2 Visitante","Over 2.5","Under 2.5"])

            filtered_vb = [v for v in all_vb
                           if v["conf_label"] in conf_filter and v["label"] in market_filter]

            if not filtered_vb:
                st.warning("Ningún pick pasa los filtros actuales.")
            else:
                st.success(f"✅ {len(filtered_vb)} value bet(s) encontrado(s)")

                for vb in filtered_vb:
                    dt       = datetime.fromisoformat(vb["date"].replace("Z","+00:00"))
                    date_str = dt.strftime("%a %d/%m · %H:%M UTC")
                    ck       = vb["conf_key"]
                    hctx     = vb["home_ctx"]
                    actx     = vb["away_ctx"]

                    # Contexto en la card
                    ctx_html = (f'<br><small>{ctx_badge(hctx)} '
                                f'<span style="opacity:.6">vs</span> '
                                f'{ctx_badge(actx)}</small>')

                    # Alertas de contexto
                    alert_html = ""
                    for a in vb["ctx_alerts"]:
                        alert_html += f'<div class="ctx-alert">{a}</div>'

                    st.markdown(f"""
                    <div class="vb-{ck}">
                      <h4>{vb['home']} vs {vb['away']}</h4>
                      <small>📅 {date_str} &nbsp;|&nbsp; Mercado: <b>{vb['label']}</b>
                      &nbsp;|&nbsp; Jornada {vb['matchday']}</small>
                      {ctx_html}
                      <br><br>
                      <span class="ev-badge badge-{ck}">EV +{vb['ev']*100:.1f}%</span>
                      <span class="conf-label">{vb['conf_icon']} Confianza {vb['conf_label']}</span>
                      &nbsp;&nbsp;
                      <span style="font-size:.88rem">
                        Cuota: <b>{vb['bk_odds']}</b> ·
                        Justa: <b>{round(1/vb['model_p'],2)}</b> ·
                        Modelo: <b>{vb['model_p']*100:.1f}%</b> ·
                        Implícita: <b>{vb['implied']*100:.1f}%</b> ·
                        Edge: <b>+{vb['edge']*100:.1f}pp</b>
                      </span>
                      {alert_html}
                    </div>
                    """, unsafe_allow_html=True)

                # Tabla
                st.divider()
                df_vb = pd.DataFrame(filtered_vb)
                df_vb["EV %"]        = (df_vb["ev"]      * 100).round(1)
                df_vb["Edge pp"]     = (df_vb["edge"]     * 100).round(1)
                df_vb["Modelo %"]    = (df_vb["model_p"]  * 100).round(1)
                df_vb["Implícita %"] = (df_vb["implied"]  * 100).round(1)
                df_vb["Ctx Local"]   = df_vb["home_ctx"].apply(lambda x: f"{x['emoji']} {x['label']}")
                df_vb["Ctx Visit."]  = df_vb["away_ctx"].apply(lambda x: f"{x['emoji']} {x['label']}")
                show = ["home","away","label","bk_odds","Modelo %","Implícita %",
                        "Edge pp","EV %","conf_label","Ctx Local","Ctx Visit."]
                st.dataframe(
                    df_vb[show].rename(columns={
                        "home":"Local","away":"Visitante","label":"Mercado",
                        "bk_odds":"Cuota","conf_label":"Confianza"
                    }),
                    use_container_width=True, hide_index=True
                )
                csv = df_vb[show].to_csv(index=False)
                st.download_button("📥 Descargar CSV", csv,
                    f"statitum_vb_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    # ─────────────────────────────────────────────────────────
    # TAB 2 · TODOS LOS PARTIDOS
    # ─────────────────────────────────────────────────────────
    with tab_all:
        st.markdown(f"### Próximos {days_ahead} días · {league_name}")
        if not upcoming:
            st.info("No hay partidos programados en este período.")
        else:
            for m in upcoming:
                info   = match_map[m["id"]]
                p      = info["p"]
                bk     = info["bk"]
                vbets  = info["vbets"]
                hctx   = info["home_ctx"]
                actx   = info["away_ctx"]
                alerts = info["alerts"]
                dt = datetime.fromisoformat(m["date"].replace("Z","+00:00"))

                title = (f"**{m['home_name']}** {hctx['emoji']} vs "
                         f"{actx['emoji']} **{m['away_name']}** · "
                         f"{dt.strftime('%a %d/%m %H:%M UTC')}")
                if vbets:
                    title += f"  🎯 {len(vbets)} pick(s)"
                if alerts:
                    title += "  ⚠️"

                with st.expander(title, expanded=False):
                    # Contexto
                    ctx_col1, ctx_col2 = st.columns(2)
                    ctx_col1.markdown(
                        f"🏠 **{m['home_name']}**<br>"
                        f'<span class="ctx-badge {hctx["css"]}">'
                        f'{hctx["emoji"]} {hctx["label"]}</span>',
                        unsafe_allow_html=True
                    )
                    ctx_col2.markdown(
                        f"✈️ **{m['away_name']}**<br>"
                        f'<span class="ctx-badge {actx["css"]}">'
                        f'{actx["emoji"]} {actx["label"]}</span>',
                        unsafe_allow_html=True
                    )

                    # Alertas
                    for a in alerts:
                        st.warning(a)

                    if not p:
                        st.info("Datos insuficientes para proyectar este partido.")
                        continue

                    # Forma
                    col1, col2, col3 = st.columns([4,1,4])
                    with col1:
                        fh = team_form(season_df, m["home_id"])
                        if fh: st.markdown(badges(fh), unsafe_allow_html=True)
                        st.caption(f"λ esperada: **{p['lam_h']}** goles")
                    with col2:
                        st.markdown("**VS**")
                    with col3:
                        fa = team_form(season_df, m["away_id"])
                        if fa: st.markdown(badges(fa), unsafe_allow_html=True)
                        st.caption(f"λ esperada: **{p['lam_a']}** goles")

                    st.markdown(match_probs_html(p, m["home_name"], m["away_name"]),
                                unsafe_allow_html=True)

                    if bk["h2h_1"] > 0:
                        st.divider()
                        b1,b2,b3,b4,b5 = st.columns(5)
                        b1.metric("🏠 Local",     bk["h2h_1"], f"Justa: {p['fair_1']}")
                        b2.metric("🤝 Empate",    bk["h2h_x"], f"Justa: {p['fair_x']}")
                        b3.metric("✈️ Visit.",    bk["h2h_2"], f"Justa: {p['fair_2']}")
                        b4.metric("Over 2.5",     bk["o25"],   f"Justa: {p['fair_o25']}")
                        b5.metric("Under 2.5",    bk["u25"],   f"Justa: {p['fair_u25']}")

                    for vb in vbets:
                        st.success(
                            f"🎯 **{vb['label']}** @ {vb['bk_odds']} "
                            f"| Modelo: {vb['model_p']*100:.1f}% "
                            f"| Implícita: {vb['implied']*100:.1f}% "
                            f"| **EV +{vb['ev']*100:.1f}%** "
                            f"| {vb['conf_icon']} {vb['conf_label']}"
                        )

    # ─────────────────────────────────────────────────────────
    # TAB 3 · ANÁLISIS DE EQUIPO
    # ─────────────────────────────────────────────────────────
    with tab_team:
        st.markdown("### Análisis de equipo")
        all_teams  = {row["home_id"]:row["home_name"] for _,row in season_df.iterrows()}
        all_teams.update({row["away_id"]:row["away_name"] for _,row in season_df.iterrows()})
        name_to_id = {v:k for k,v in all_teams.items()}
        sel_name   = st.selectbox("Selecciona un equipo", sorted(all_teams.values()))
        sel_id     = name_to_id[sel_name]

        # Contexto en tabla
        if not standings_df.empty:
            last_md = int(standings_df["played"].max()) if not standings_df.empty else 0
            sel_ctx = get_team_context(sel_id, standings_df, league_cfg, last_md)
            st.markdown(
                f'<span class="ctx-badge {sel_ctx["css"]}" style="font-size:1rem;padding:5px 14px">'
                f'{sel_ctx["emoji"]} {sel_ctx["label"]}</span>',
                unsafe_allow_html=True
            )
            st.caption(" ")

        if sel_id in ratings:
            r = ratings[sel_id]
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Partidos jugados",    r["n"])
            m2.metric("N efectivo (pesos)", r["n_eff"])
            m3.metric("Goles anotados/pj",  r["gs_avg"])
            m4.metric("Goles recibidos/pj", r["gc_avg"])
            m5.metric("Diferencia/pj", f"{round(r['gs_avg']-r['gc_avg'],2):+.2f}")

            st.divider()
            st.markdown("**Últimos 8 resultados**")
            f = team_form(season_df, sel_id, 8)
            if f: st.markdown(badges(f), unsafe_allow_html=True)

            st.divider()
            st.markdown("**Ratings calibrados (1.00 = media de liga)**")
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Ataque local",      round(r["att_h"],3), "↑" if r["att_h"]>1 else "↓")
            r2.metric("Ataque visitante",  round(r["att_a"],3), "↑" if r["att_a"]>1 else "↓")
            r3.metric("Defensa local",     round(r["def_h"],3), "↓ sólida" if r["def_h"]<1 else "↑ porosa")
            r4.metric("Defensa visitante", round(r["def_a"],3), "↓ sólida" if r["def_a"]<1 else "↑ porosa")

            st.divider()
            hm_t = season_df[season_df["home_id"]==sel_id][["date","home_goals","away_goals"]].rename(columns={"home_goals":"gf","away_goals":"gc"})
            am_t = season_df[season_df["away_id"]==sel_id][["date","home_goals","away_goals"]].rename(columns={"away_goals":"gf","home_goals":"gc"})
            all_m = pd.concat([hm_t, am_t]).sort_values("date").reset_index(drop=True)
            if not all_m.empty:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=all_m.index, y=all_m["gf"], name="Anotados",
                    line=dict(color="#2d6a4f",width=2), mode="lines+markers"))
                fig.add_trace(go.Scatter(x=all_m.index, y=all_m["gc"], name="Recibidos",
                    line=dict(color="#a4262c",width=2), mode="lines+markers"))
                fig.add_hline(y=r["gs_avg"], line_dash="dot", line_color="#2d6a4f")
                fig.add_hline(y=r["gc_avg"], line_dash="dot", line_color="#a4262c")
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(245,245,245,0.4)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para este equipo.")

    # ─────────────────────────────────────────────────────────
    # TAB 4 · CLASIFICACIÓN
    # ─────────────────────────────────────────────────────────
    with tab_table:
        st.markdown(f"### Clasificación actual · {league_name}")
        if standings_df.empty:
            st.warning("No se pudo cargar la clasificación.")
        else:
            # Agregar contexto a cada equipo
            last_md = int(standings_df["played"].max())
            def ctx_for_row(row):
                c = get_team_context(row["team_id"], standings_df, league_cfg, last_md)
                return f"{c['emoji']} {c['label']}"
            standings_df["Situación"] = standings_df.apply(ctx_for_row, axis=1)
            standings_df["gf"] = standings_df["gf"].astype(int)
            standings_df["ga"] = standings_df["ga"].astype(int)
            standings_df["gd"] = standings_df["gd"].apply(lambda x: f"{int(x):+d}")

            st.dataframe(
                standings_df.rename(columns={
                    "position":"Pos","team_name":"Equipo","played":"PJ",
                    "points":"Pts","gf":"GF","ga":"GC","gd":"DG"
                })[["Pos","Equipo","PJ","Pts","GF","GC","DG","Situación"]],
                use_container_width=True, hide_index=True
            )

    st.markdown("""
    <div class="footer">
      <b>Statitum</b> · Poisson Calibrado + Contexto Competitivo · football-data.org · the-odds-api.com<br>
      Las probabilidades son estimaciones estadísticas. Apuesta de forma responsable.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
