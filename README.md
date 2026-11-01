# Bond Yield Rich/Cheap Model

A Streamlit web application that uses **rolling multiple linear regression (OLS)** to determine whether US and Australian 10-year government bond yields are **RICH** (too low vs fundamentals), **CHEAP** (too high), or **NEUTRAL**.

## Quick Start

1. Upload your Excel file containing two tabs of monthly data
2. Select the country (US 10Y or AU 10Y) from the sidebar
3. The model runs automatically and displays the current signal

## How It Works

The model regresses bond yields against macro and market factors using a rolling window (default 60 months). The residual (actual yield minus predicted "fair value" yield) is normalised into a Z-Score. Yields significantly above fair value → **CHEAP** (bonds undervalued, buy signal). Yields significantly below fair value → **RICH** (bonds overvalued, sell/hedge signal).

## Input File Format

A single `.xlsx` file with **two tabs**:

- **`US Model field and data`** — US 10Y yield and its independent variables
- **`AU Model field and data`** — AU 10Y yield and its independent variables

Each tab has the same layout:

| Row | Content |
|-----|---------|
| Row 1 | Column headers (Cell A1 may contain a Bloomberg reference date — ignored) |
| Row 2 | Feature flags: `L`, `LM`, `LMP`, `MP`, `P`, or `N` (blank for Date and yield columns) |
| Row 3+ | Monthly data |

### Feature Flags

| Flag | Features Generated |
|------|--------------------|
| `L`  | Level only |
| `LM` | Level + Month-over-Month Change |
| `LMP`| Level + MoM Change + % Change |
| `MP` | MoM Change + % Change |
| `P`  | % Change only |
| `N`  | Exclude from model |

### Signal Interpretation (Bonds)

- **CHEAP:** Yields are *higher* than fair value → bonds are undervalued → buying opportunity
- **RICH:** Yields are *lower* than fair value → bonds are overvalued → sell/hedge
- **NEUTRAL:** Yields are near fair value

Uses **symmetric thresholds** (default 1.5σ, adjustable via slider).

## Monthly Commentary

The Commentary & Attribution box explains the change in the model's fair-value yield. Its lines are:

| Line | Meaning |
|------|---------|
| **Move vs published prior month** | Current fair value minus last month's fair value as you published it. Shown only when a published figure is available. |
| **Prior-month revision** | Last month's fair value as restated by this upload, minus the figure you published. Non-zero when last month's inputs changed (for example macro data released after the previous download). Shown only when a published figure is available. |
| **Move vs restated prior month** | Current fair value minus last month's fair value as restated by this upload. The factor lines below explain this move, and it equals the sum of all rows in the attribution table. |
| **Stale inputs** | Active inputs whose latest value is blank and carried forward, or identical to the prior month. The latest fair value relies on these and may be restated next month. A genuinely unchanged input (for example Fed Funds) is also flagged, so check before concluding data is stale. |
| **Primary / Secondary / Tertiary Driver** | Largest factor contributions to the move vs restated prior month. |
| **Model Recalibration** | Part of the move caused by the rolling window shifting (coefficient drift and intercept change). Included in the move vs restated prior month. |
| **Signal Status** | Current signal, Z-Score and in-sample R². |

Move vs published prior month = Prior-month revision + Move vs restated prior month.

### Published history

The app does not store anything between uploads, so it needs last month's published fair value to show the revision. Either:

- Type it in the sidebar under **Published history** ("last month's published fair value", e.g. `5.02`), or
- Keep a `published_fair_value.csv` in the model folder. Use **Download updated published history (CSV)** (or **Save to local history file** when running locally) after you publish each month, then upload the CSV in the sidebar next month. When running locally, the app reads the file from the model folder automatically.

Streamlit Cloud has no persistent disk, so the CSV must be kept by you and re-uploaded. History rows are keyed by country, model specification, lookback and date.

## Deployment

Deployed on [Streamlit Cloud](https://share.streamlit.io). Push changes to `main` branch and Streamlit auto-deploys.

## Dependencies

See `requirements.txt`. All standard Python packages — no Bloomberg API dependency (data is uploaded via Excel).
