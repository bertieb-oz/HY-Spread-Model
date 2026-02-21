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
| **Input** | Monthly Excel file — any number of independent variables with Row 2 feature flags |
| **Feature Flags** | `L` (level), `LM` (level + MoM change), `LMP` (level + MoM + %), `MP` (MoM + %), `P` (% only), `N` (exclude) |
| **Auto-Selection** | For `LMP`/`MP` flags, the model auto-selects MoM or % change per rolling window (eliminates multicollinearity) |
| **Model** | Rolling OLS regression (36/48/60-month window) |
| **Signal** | Z-score of residuals with asymmetric thresholds (default: CHEAP at 1.0σ, RICH at 2.0σ) |

### Sidebar

The sidebar displays model settings, detected variables with their flags, per-window feature selection results (which transformation was chosen for competing MoM/% pairs), and a coefficient summary for the latest rolling window.

## Expected Excel Format

```
| Date       | US High Yield Spread | VIX  | Oil   | CDX HY | MOVE | SPX  | HYG Shares | US 10Y Yield |
|            |                      | LM   | LM    | MP     | LM   | LM   | P          | L            |
| 2020-01-31 | 3.56                 | 18.8 | 51.56 | 320    | 57.8 | 3230 | 250000     | 1.51         |
| 2020-02-28 | 4.01                 | 40.1 | 44.76 | 380    | 72.3 | 2954 | 280000     | 1.15         |
```

- **Row 1:** Column headers (Date and HY Spread are auto-detected by name)
- **Row 2:** Feature flags — blank for Date and HY Spread, a valid flag for each independent variable
- **Row 3+:** Monthly data
- Add or remove columns freely — no code changes needed
