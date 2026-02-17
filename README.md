# HY Spread Rich/Cheap Model

A Streamlit web app that uses rolling multiple linear regression to determine if US High Yield credit spreads are **rich** (too tight), **cheap** (too wide), or **neutral** relative to fundamental market drivers.

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploy to Streamlit Cloud (Free)

1. **Create a GitHub repo** — push `app.py`, `requirements.txt`, and this README.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. You'll get a public URL like `https://your-app.streamlit.app`.

## How It Works

| Step | Detail |
|------|--------|
| **Input** | Monthly Excel file with HY Spread, VIX, Oil, CDX, MOVE, SPX, HYG Shares |
| **Features** | 10 variables: levels + month-over-month changes |
| **Model** | Rolling OLS regression (36/48/60-month window) |
| **Signal** | Z-score of residuals with asymmetric thresholds |

### Why Asymmetric Thresholds?

Selling HY bonds loses 4–6% annual carry. The model requires a **stronger** signal to say RICH/sell (default 2.0σ) than CHEAP/buy (default 1.0σ).

## Expected Excel Format

| Date | US High Yield Spread | VIX | Oil | CDX | MOVE | SPX Yield | HYG Equity Outstanding |
|------|---------------------|-----|-----|-----|------|-----------|----------------------|
| 2015-12-30 | 6.58 | 17.29 | 37.04 | 348.4 | 67.35 | 2063.36 | 180600006 |

The app auto-detects common column name variations and skips Bloomberg ticker sub-headers.
