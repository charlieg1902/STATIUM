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
  .value-card {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    border-radius: 12px; padding: 18px; color: white; margin-bottom: 12px;
  }
  .value-card h4 { margin: 0 0 6px 0; font-size: 1.1rem; }
  .ev-badge {
    background: #52b788; color: white; font-weight: 700;
    padding: 4px 12px; border-radius: 20px; font-size: 1rem;
  }
  .neutral-card {
    background: #f8f9fa; border: 1px solid #dee2e6;
    border-radius: 10px; padding: 14px; margin-bottom: 10px;
  }
  .prob-row { display: flex; gap: 8px; margin-top: 8px; }
  .prob-box {
    flex: 1; text-align: center; border-radius: 8px; padding: 10px 4px;
    font-weight: 600; font-size: 0.9rem;
  }
  .home-box  { background: #d8f3dc; color: #1b4332; }
  .draw-box  { background: #fff3cd; color: #664d03; }
  .away-box  { background: #d0e8f7; color: #0a3d62; }
  .over-box  { background: #fde8d8; color: #7f3f00; }
  .under-box { background: #e8d8fd; color: #3f007f; }
  .btts-box  { background: #d8eafd; color: #003f7f; }
  div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
  .footer { text-align:center; color:#aaa; font-size:0.78rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
FD_BASE   = "https://api.football-data.org/v4"
ODDS_BASE = "https://api.the-odds-api.com/v4"

LEAGUES = {
    "🇬🇧 Premier League":  {"fd": "PL",  "odds": "soccer_epl"},
    "🇪🇸 La Liga":          {"fd": "PD",  "odds": "soccer_spain_la_liga"},
    "🇮🇹 Serie A":          {"fd": "SA",  "odds": "soccer_italy_serie_a"},
    "🇩🇪 Bundesliga":       {"fd": "BL1", "odds": "soccer_germany_bundesliga"},
    "🇫🇷 Ligue 1":          {"fd": "FL1", "odds": "soccer_france_ligue_one"},
}

VALUE_THRESHOLD = 0.04   # Minimum EV to flag as value bet (4%)
MIN_ODDS       = 1.30    # Ignore odds below this (too short to be interesting)
MAX_ODDS       = 8.00    # Ignore odds above this (too risky / low liquidity)

# ═══════════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════════

def _fd_get(api_key, endpoint, params=None):
    try:
        r = requests.get(
            f"{FD_BASE}{endpoint}",
            headers={"X-Auth-Token": api_key},
            params=params, timeout=12
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def _odds_get(api_key, endpoint, params=None):
    try:
        r = requests.get(
            f"{ODDS_BASE}{endpoint}",
            params={"apiKey": api_key, **(params or {})},
            timeout=12
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

# ─── Cached fetchers ───────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_season_matches(fd_key, fd_code):
    """All finished matches of current season (used to build team ratings)."""
    data = _fd_get(fd_key, f"/competitions/{fd_code}/matches", {"status": "FINISHED"})
    if not data:
        return pd.DataFrame()
    rows = []
    for m in data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        if ft.get("home") is not None:
            rows.append({
                "date":      m["utcDate"],
                "home_id":   m["homeTeam"]["id"],
                "home_name": m["homeTeam"]["name"],
                "away_id":   m["awayTeam"]["id"],
                "away_name": m["awayTeam"]["name"],
                "home_goals": ft["home"],
                "away_goals": ft["away"],
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_upcoming_matches(fd_key, fd_code, days=7):
    """Scheduled matches in the next N days."""
    today  = datetime.utcnow().strftime("%Y-%m-%d")
    future = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
    data = _fd_get(fd_key, f"/competitions/{fd_code}/matches",
                   {"status": "SCHEDULED", "dateFrom": today, "dateTo": future})
    if not data:
        return []
    out = []
    for m in data.get("matches", []):
        out.append({
            "id":        m["id"],
            "date":      m["utcDate"],
            "home_id":   m["homeTeam"]["id"],
            "home_name": m["homeTeam"]["name"],
            "away_id":   m["awayTeam"]["id"],
            "away_name": m["awayTeam"]["name"],
            "matchday":  m.get("matchday", ""),
        })
    return out

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_odds(odds_key, sport_key):
    """Live odds from the-odds-api (cached 2 h to preserve monthly quota)."""
    data = _odds_get(
        odds_key,
        f"/sports/{sport_key}/odds/",
        {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
    )
    return data if isinstance(data, list) else []

# ═══════════════════════════════════════════════════════════
# POISSON MODEL
# ═══════════════════════════════════════════════════════════

def build_ratings(df):
    """Compute attack / defense ratings for each team (season data)."""
    if df.empty or len(df) < 20:
        return {}, 1.35, 1.10

    avg_h = df["home_goals"].mean()
    avg_a = df["away_goals"].mean()
    if avg_h == 0 or avg_a == 0:
        return {}, 1.35, 1.10

    ratings = {}
    for tid in set(df["home_id"]) | set(df["away_id"]):
        hm = df[df["home_id"] == tid]
        am = df[df["away_id"] == tid]
        nh, na = len(hm), len(am)
        if nh + na < 4:
            continue
        att_h = hm["home_goals"].mean() / avg_h if nh >= 2 else 1.0
        att_a = am["away_goals"].mean() / avg_a if na >= 2 else 1.0
        def_h = hm["away_goals"].mean() / avg_a if nh >= 2 else 1.0
        def_a = am["home_goals"].mean() / avg_h if na >= 2 else 1.0
        gs_avg = (hm["home_goals"].sum() + am["away_goals"].sum()) / (nh + na)
        gc_avg = (hm["away_goals"].sum() + am["home_goals"].sum()) / (nh + na)
        ratings[tid] = {
            "att_h": att_h, "att_a": att_a,
            "def_h": def_h, "def_a": def_a,
            "gs_avg": round(gs_avg, 2), "gc_avg": round(gc_avg, 2),
            "n": nh + na,
        }
    return ratings, avg_h, avg_a


def match_probs(home_id, away_id, ratings, avg_h, avg_a):
    """Bivariate Poisson probabilities for a match."""
    if home_id not in ratings or away_id not in ratings:
        return None
    hr, ar = ratings[home_id], ratings[away_id]

    lam_h = np.clip(hr["att_h"] * ar["def_a"] * avg_h, 0.30, 5.0)
    lam_a = np.clip(ar["att_a"] * hr["def_h"] * avg_a, 0.30, 5.0)

    G = 8  # score matrix dimension
    ph = [poisson.pmf(i, lam_h) for i in range(G)]
    pa = [poisson.pmf(i, lam_a) for i in range(G)]
    M  = np.outer(ph, pa)

    home_win = float(np.sum(np.tril(M, -1)))
    draw     = float(np.sum(np.diag(M)))
    away_win = float(np.sum(np.triu(M, 1)))

    over25  = sum(M[i][j] for i in range(G) for j in range(G) if i + j > 2)
    under25 = 1.0 - over25
    btts    = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))

    def fair(p): return round(1 / p, 2) if p > 0.01 else 99.0

    return {
        "lam_h": round(lam_h, 2), "lam_a": round(lam_a, 2),
        "home_win": round(home_win, 4),
        "draw":     round(draw, 4),
        "away_win": round(away_win, 4),
        "over25":   round(over25, 4),
        "under25":  round(under25, 4),
        "btts":     round(btts, 4),
        "fair_1":   fair(home_win),
        "fair_x":   fair(draw),
        "fair_2":   fair(away_win),
        "fair_o25": fair(over25),
        "fair_u25": fair(under25),
        "fair_btts":fair(btts),
    }


def team_form(df, team_id, n=6):
    hm = df[df["home_id"] == team_id][["date","home_goals","away_goals"]].rename(
        columns={"home_goals":"gf","away_goals":"gc"})
    hm["v"] = "H"
    am = df[df["away_id"] == team_id][["date","home_goals","away_goals"]].rename(
        columns={"away_goals":"gf","home_goals":"gc"})
    am["v"] = "A"
    combined = pd.concat([hm, am]).sort_values("date", ascending=False).head(n)
    result = []
    for _, r in combined.iterrows():
        if   r.gf > r.gc: result.append(("W", "#2d6a4f", r.gf, r.gc, r.v))
        elif r.gf == r.gc: result.append(("D", "#d4a017", r.gf, r.gc, r.v))
        else:              result.append(("L", "#a4262c", r.gf, r.gc, r.v))
    return result

# ═══════════════════════════════════════════════════════════
# VALUE BET ENGINE
# ═══════════════════════════════════════════════════════════

def _sim(a, b):
    a = a.lower().replace(" fc","").replace(" cf","").replace(" afc","").replace(" sc","").strip()
    b = b.lower().replace(" fc","").replace(" cf","").replace(" afc","").replace(" sc","").strip()
    return SequenceMatcher(None, a, b).ratio()

def find_odds_match(fd_home, fd_away, fd_date_str, odds_list):
    """Fuzzy-match a football-data match to an odds-api match."""
    fd_date = datetime.fromisoformat(fd_date_str.replace("Z", "+00:00")).date()
    best, best_score = None, 0
    for om in odds_list:
        try:
            om_date = datetime.fromisoformat(
                om["commence_time"].replace("Z", "+00:00")).date()
        except Exception:
            continue
        if abs((om_date - fd_date).days) > 1:
            continue
        score = _sim(fd_home, om["home_team"]) + _sim(fd_away, om["away_team"])
        if score > best_score and score > 1.25:
            best_score, best = score, om
    return best

def best_odds_for(om):
    """Extract best available decimal odds across bookmakers."""
    best = {"h2h_1": 0, "h2h_x": 0, "h2h_2": 0, "o25": 0, "u25": 0}
    if not om:
        return best
    home_t = om.get("home_team", "")
    away_t = om.get("away_team", "")
    for bk in om.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                for oc in mkt.get("outcomes", []):
                    p = float(oc.get("price", 0))
                    n = oc.get("name", "")
                    if _sim(n, home_t) > 0.7:
                        best["h2h_1"] = max(best["h2h_1"], p)
                    elif _sim(n, away_t) > 0.7:
                        best["h2h_2"] = max(best["h2h_2"], p)
                    elif "draw" in n.lower():
                        best["h2h_x"] = max(best["h2h_x"], p)
            elif mkt["key"] == "totals":
                for oc in mkt.get("outcomes", []):
                    pt = float(oc.get("point", 0))
                    pr = float(oc.get("price", 0))
                    if abs(pt - 2.5) < 0.01:
                        if oc.get("name","").lower() == "over":
                            best["o25"] = max(best["o25"], pr)
                        elif oc.get("name","").lower() == "under":
                            best["u25"] = max(best["u25"], pr)
    return best

def compute_ev(model_prob, bookie_odds):
    """Expected Value = model_prob * odds - 1."""
    if bookie_odds <= 1.0 or model_prob <= 0:
        return None
    return round(model_prob * bookie_odds - 1, 4)

def detect_value_bets(probs, bk_odds, home_name, away_name):
    """Return list of value bet opportunities for a match."""
    checks = [
        ("1 Local",      probs["home_win"], bk_odds["h2h_1"]),
        ("X Empate",     probs["draw"],     bk_odds["h2h_x"]),
        ("2 Visitante",  probs["away_win"], bk_odds["h2h_2"]),
        ("Over 2.5",     probs["over25"],   bk_odds["o25"]),
        ("Under 2.5",    probs["under25"],  bk_odds["u25"]),
    ]
    found = []
    for label, model_p, bk_odd in checks:
        if bk_odd < MIN_ODDS or bk_odd > MAX_ODDS:
            continue
        ev = compute_ev(model_p, bk_odd)
        if ev is not None and ev >= VALUE_THRESHOLD:
            implied = round(1 / bk_odd, 4)
            edge    = round(model_p - implied, 4)
            found.append({
                "label":    label,
                "model_p":  model_p,
                "implied":  implied,
                "edge":     edge,
                "bk_odds":  bk_odd,
                "ev":       ev,
                "home":     home_name,
                "away":     away_name,
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
                 f' title="{gf}-{gc} ({v})">{res}</span>')
    return html

def prob_html(label, value, css_class):
    return (f'<div class="prob-box {css_class}">'
            f'{label}<br><b>{value*100:.1f}%</b></div>')

def match_probs_html(p, home_name, away_name):
    row1 = (prob_html(f"🏠 {home_name[:12]}", p["home_win"], "home-box") +
            prob_html("🤝 Empate",             p["draw"],     "draw-box") +
            prob_html(f"✈️ {away_name[:12]}",  p["away_win"], "away-box"))
    row2 = (prob_html("Over 2.5", p["over25"], "over-box") +
            prob_html("Under 2.5", p["under25"], "under-box") +
            prob_html("BTTS", p["btts"], "btts-box"))
    return (f'<div class="prob-row">{row1}</div>'
            f'<div class="prob-row" style="margin-top:6px">{row2}</div>')

# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════

def main():
    # ── Header ──────────────────────────────────────────────
    st.markdown("## 📊 Statitum")
    st.caption("Análisis estadístico deportivo · Detección de value bets con EV real")
    st.divider()

    # ── Secrets ─────────────────────────────────────────────
    try:
        FD_KEY   = st.secrets["FOOTBALL_API_KEY"]
        ODDS_KEY = st.secrets["ODDS_API_KEY"]
    except Exception:
        st.error("⚠️ Faltan las API Keys. Ve a **Settings → Secrets** en Streamlit Cloud y agrega:\n"
                 "```\nFOOTBALL_API_KEY = \"...\"\nODDS_API_KEY = \"...\"\n```")
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        league_name = st.selectbox("Liga", list(LEAGUES.keys()))
        fd_code    = LEAGUES[league_name]["fd"]
        odds_sport = LEAGUES[league_name]["odds"]
        days_ahead = st.slider("Próximos días", 1, 14, 7)
        ev_min_pct = st.slider("EV mínimo para value bet (%)", 2, 15, 4)
        global VALUE_THRESHOLD
        VALUE_THRESHOLD = ev_min_pct / 100

        st.divider()
        st.markdown("### ℹ️ Modelo")
        st.info("**Poisson Bivariado**\n\nCalcula goles esperados (λ) por equipo basado "
                "en sus ratings de ataque y defensa relativos a la media de la liga.")
        st.divider()
        if st.button("🔄 Limpiar caché"):
            st.cache_data.clear()
            st.rerun()

    # ── Load data ────────────────────────────────────────────
    with st.spinner("📡 Cargando datos de la temporada..."):
        season_df = fetch_season_matches(FD_KEY, fd_code)

    if season_df.empty:
        st.error("No se pudieron cargar datos. Verifica tu FOOTBALL_API_KEY.")
        st.stop()

    ratings, avg_h, avg_a = build_ratings(season_df)

    with st.spinner("🗓️ Cargando próximos partidos..."):
        upcoming = fetch_upcoming_matches(FD_KEY, fd_code, days_ahead)

    with st.spinner("💰 Cargando cuotas del mercado..."):
        odds_list = fetch_odds(ODDS_KEY, odds_sport)

    # ── Stats bar ────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Partidos históricos", len(season_df))
    with col2: st.metric("Equipos con rating", len(ratings))
    with col3: st.metric("Próximos partidos", len(upcoming))
    with col4: st.metric("Mercados con cuotas", len(odds_list))
    st.divider()

    # ── Tabs ─────────────────────────────────────────────────
    tab_vb, tab_all, tab_team = st.tabs([
        "🎯 Value Bets", "🗓️ Todos los Partidos", "🔍 Análisis de Equipo"
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1 · VALUE BETS
    # ─────────────────────────────────────────────────────────
    with tab_vb:
        st.markdown(f"### Value Bets detectados · {league_name}")
        st.caption(f"Mínimo EV requerido: **{ev_min_pct}%** · Cuotas: {MIN_ODDS} – {MAX_ODDS}")

        all_vb = []
        for m in upcoming:
            p = match_probs(m["home_id"], m["away_id"], ratings, avg_h, avg_a)
            if not p:
                continue
            om    = find_odds_match(m["home_name"], m["away_name"], m["date"], odds_list)
            bk    = best_odds_for(om)
            vbets = detect_value_bets(p, bk, m["home_name"], m["away_name"])
            for vb in vbets:
                vb["date"] = m["date"]
                all_vb.append(vb)

        if not all_vb:
            st.info("No se detectaron value bets con los criterios actuales. "
                    "Prueba bajando el EV mínimo o ampliando el rango de días.")
        else:
            # Sort by EV descending
            all_vb.sort(key=lambda x: x["ev"], reverse=True)
            st.success(f"✅ {len(all_vb)} value bet(s) encontrado(s)")

            for vb in all_vb:
                match_dt = datetime.fromisoformat(vb["date"].replace("Z", "+00:00"))
                date_str = match_dt.strftime("%a %d/%m · %H:%M UTC")
                ev_pct   = vb["ev"] * 100
                edge_pct = vb["edge"] * 100

                st.markdown(f"""
                <div class="value-card">
                  <h4>{vb['home']} vs {vb['away']}</h4>
                  <small>📅 {date_str} &nbsp;|&nbsp; Mercado: <b>{vb['label']}</b></small><br><br>
                  <span class="ev-badge">EV +{ev_pct:.1f}%</span>
                  &nbsp;&nbsp;
                  <span style="font-size:0.9rem">
                    Cuota disponible: <b>{vb['bk_odds']}</b> &nbsp;|&nbsp;
                    Cuota justa: <b>{round(1/vb['model_p'],2)}</b> &nbsp;|&nbsp;
                    P. modelo: <b>{vb['model_p']*100:.1f}%</b> &nbsp;|&nbsp;
                    P. implícita: <b>{vb['implied']*100:.1f}%</b> &nbsp;|&nbsp;
                    Edge: <b>+{edge_pct:.1f}%</b>
                  </span>
                </div>
                """, unsafe_allow_html=True)

            # Summary table
            st.divider()
            st.markdown("**Resumen exportable**")
            df_vb = pd.DataFrame(all_vb)
            df_vb["ev_pct"]   = (df_vb["ev"] * 100).round(1)
            df_vb["edge_pct"] = (df_vb["edge"] * 100).round(1)
            df_vb["modelo_%"] = (df_vb["model_p"] * 100).round(1)
            df_vb["implícita_%"] = (df_vb["implied"] * 100).round(1)
            show_cols = ["home","away","label","bk_odds","modelo_%","implícita_%","edge_pct","ev_pct"]
            st.dataframe(df_vb[show_cols].rename(columns={
                "home":"Local","away":"Visitante","label":"Mercado",
                "bk_odds":"Cuota","edge_pct":"Edge %","ev_pct":"EV %"
            }), use_container_width=True, hide_index=True)
            csv = df_vb[show_cols].to_csv(index=False)
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
            table_rows = []
            for m in upcoming:
                p = match_probs(m["home_id"], m["away_id"], ratings, avg_h, avg_a)
                dt = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
                om = find_odds_match(m["home_name"], m["away_name"], m["date"], odds_list)
                bk = best_odds_for(om)

                if p:
                    vbets = detect_value_bets(p, bk, m["home_name"], m["away_name"])
                    vb_labels = " · ".join(v["label"] + f" EV+{v['ev']*100:.0f}%" for v in vbets)
                    table_rows.append({
                        "Fecha":    dt.strftime("%d/%m %H:%M"),
                        "Local":    m["home_name"],
                        "Visit.":   m["away_name"],
                        "λ Local":  p["lam_h"],
                        "λ Visit.": p["lam_a"],
                        "% 1":  f"{p['home_win']*100:.0f}%",
                        "% X":  f"{p['draw']*100:.0f}%",
                        "% 2":  f"{p['away_win']*100:.0f}%",
                        "O2.5": f"{p['over25']*100:.0f}%",
                        "BTTS": f"{p['btts']*100:.0f}%",
                        "Value Bets": vb_labels if vb_labels else "—",
                    })

                with st.expander(
                    f"**{m['home_name']}** vs **{m['away_name']}** · {dt.strftime('%a %d/%m %H:%M UTC')}",
                    expanded=False
                ):
                    if not p:
                        st.warning("Datos insuficientes para proyectar este partido.")
                        continue

                    c1, c2, c3 = st.columns([4, 1, 4])
                    with c1:
                        st.markdown(f"**🏠 {m['home_name']}**")
                        form_h = team_form(season_df, m["home_id"])
                        if form_h:
                            st.markdown(badges(form_h), unsafe_allow_html=True)
                        st.caption(f"λ esperada: {p['lam_h']} goles")
                    with c2:
                        st.markdown("<br><b>VS</b>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"**✈️ {m['away_name']}**")
                        form_a = team_form(season_df, m["away_id"])
                        if form_a:
                            st.markdown(badges(form_a), unsafe_allow_html=True)
                        st.caption(f"λ esperada: {p['lam_a']} goles")

                    st.markdown(
                        match_probs_html(p, m["home_name"], m["away_name"]),
                        unsafe_allow_html=True
                    )

                    if bk["h2h_1"] > 0:
                        st.divider()
                        st.markdown("**Mejores cuotas disponibles**")
                        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                        bc1.metric("🏠 Local",    bk["h2h_1"], f"Justa: {p['fair_1']}")
                        bc2.metric("🤝 Empate",   bk["h2h_x"], f"Justa: {p['fair_x']}")
                        bc3.metric("✈️ Visitante", bk["h2h_2"], f"Justa: {p['fair_2']}")
                        bc4.metric("Over 2.5",    bk["o25"],   f"Justa: {p['fair_o25']}")
                        bc5.metric("Under 2.5",   bk["u25"],   f"Justa: {p['fair_u25']}")

                    vbets_match = detect_value_bets(p, bk, m["home_name"], m["away_name"])
                    if vbets_match:
                        st.divider()
                        for vb in vbets_match:
                            st.success(
                                f"🎯 **Value Bet** · {vb['label']} @ {vb['bk_odds']} "
                                f"| Modelo: {vb['model_p']*100:.1f}% vs Implícita: {vb['implied']*100:.1f}% "
                                f"| **EV +{vb['ev']*100:.1f}%**"
                            )

            if table_rows:
                st.divider()
                st.markdown("**Vista de tabla**")
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    # TAB 3 · ANÁLISIS DE EQUIPO
    # ─────────────────────────────────────────────────────────
    with tab_team:
        st.markdown("### Análisis de equipo")

        all_teams = {}
        for _, row in season_df.iterrows():
            all_teams[row["home_id"]] = row["home_name"]
            all_teams[row["away_id"]] = row["away_name"]

        sorted_names = sorted(all_teams.values())
        name_to_id   = {v: k for k, v in all_teams.items()}

        sel_name = st.selectbox("Selecciona un equipo", sorted_names)
        sel_id   = name_to_id[sel_name]

        if sel_id in ratings:
            r = ratings[sel_id]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Partidos jugados",    r["n"])
            m2.metric("Goles anotados/pj",   r["gs_avg"])
            m3.metric("Goles recibidos/pj",  r["gc_avg"])
            diff = round(r["gs_avg"] - r["gc_avg"], 2)
            m4.metric("Diferencia/pj", f"{diff:+.2f}")

            st.divider()
            st.markdown("**Últimos 8 resultados**")
            f = team_form(season_df, sel_id, 8)
            if f: st.markdown(badges(f), unsafe_allow_html=True)

            st.divider()
            st.markdown("**Ratings relativos a la liga (1.00 = promedio)**")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Ataque local",   round(r["att_h"], 2),
                       "↑ sobre media" if r["att_h"] > 1 else "↓ bajo media")
            rc2.metric("Ataque visitante", round(r["att_a"], 2),
                       "↑ sobre media" if r["att_a"] > 1 else "↓ bajo media")
            rc3.metric("Defensa local",  round(r["def_h"], 2),
                       "↓ sólida" if r["def_h"] < 1 else "↑ porosa")
            rc4.metric("Defensa visitante", round(r["def_a"], 2),
                       "↓ sólida" if r["def_a"] < 1 else "↑ porosa")

            st.divider()
            st.markdown("**Histórico de goles esta temporada**")
            hm_t = season_df[season_df["home_id"] == sel_id][["date","home_goals","away_goals"]].rename(
                columns={"home_goals":"gf","away_goals":"gc"})
            am_t = season_df[season_df["away_id"] == sel_id][["date","home_goals","away_goals"]].rename(
                columns={"away_goals":"gf","home_goals":"gc"})
            all_m = pd.concat([hm_t, am_t]).sort_values("date").reset_index(drop=True)

            if not all_m.empty:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=all_m.index, y=all_m["gf"],
                    name="Goles anotados", line=dict(color="#2d6a4f", width=2),
                    mode="lines+markers"))
                fig.add_trace(go.Scatter(x=all_m.index, y=all_m["gc"],
                    name="Goles recibidos", line=dict(color="#a4262c", width=2),
                    mode="lines+markers"))
                fig.add_hline(y=r["gs_avg"], line_dash="dot",
                              line_color="#2d6a4f", annotation_text="Media anotados")
                fig.add_hline(y=r["gc_avg"], line_dash="dot",
                              line_color="#a4262c", annotation_text="Media recibidos")
                fig.update_layout(
                    height=300, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(245,245,245,0.5)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para este equipo esta temporada.")

    # ── Footer ───────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
      <b>Statitum</b> · Modelo Poisson Bivariado · Datos: football-data.org + the-odds-api.com<br>
      Las probabilidades son estimaciones estadísticas, no garantías. Apuesta de forma responsable.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
