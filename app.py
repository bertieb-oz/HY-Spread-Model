"""
US High Yield Spread Rich/Cheap Model
======================================
A Streamlit web application that uses rolling multiple linear regression
to determine if US High Yield credit spreads are rich, cheap, or neutral
relative to fundamental market drivers.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import io
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HY Spread Rich/Cheap Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .signal-box {
        padding: 24px; border-radius: 12px; text-align: center;
        margin-bottom: 16px; color: white; font-weight: bold;
    }
    .signal-cheap { background: linear-gradient(135deg, #1b8a3d, #27ae60); }
    .signal-rich  { background: linear-gradient(135deg, #c0392b, #e74c3c); }
    .signal-neutral { background: linear-gradient(135deg, #636e72, #95a5a6); }
    div[data-testid="stSidebar"] { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Data loading & Logic
# ─────────────────────────────────────────────────────────────

def load_data(uploaded_file):
    try:
        raw = pd.read_excel(uploaded_file, header=None)
        headers = [str(h).strip() for h in raw.iloc[0]]
        flag_row = [str(f).strip().upper() for f in raw.iloc[1]]
        df = raw.iloc[2:].reset_index(drop=True)
        df.columns = headers

        # Basic auto-detection
        date_col = next((c for c in headers if "date" in c.lower()), headers[0])
        hy_col = next((c for c in headers if any(a in c.lower() for a in ["hy spread", "high yield", "oas"])), None)

        variable_flags = {}
        original_names = {}
        for i, col in enumerate(headers):
            if col in [date_col, hy_col]: continue
            variable_flags[col.lower().replace(" ", "_")] = flag_row[i] if flag_row[i] in ["L", "LM", "LMP", "MP", "P", "N"] else "LM"
            original_names[col.lower().replace(" ", "_")] = col

        df = df.rename(columns={date_col: "date", hy_col: "hy_spread", **{orig: k for k, orig in original_names.items()}})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in df.columns:
            if c != "date": df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # Scaling HYG
        if "hyg_shares_outstanding" in df.columns:
            df["hyg_shares_outstanding"] = df["hyg_shares_outstanding"] / 1_000_000

        return df.dropna(subset=["date"]).sort_values("date").ffill(), hy_col, variable_flags, original_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def engineer_features(df, variable_flags, original_names):
    feat = df[["date", "hy_spread"]].copy()
    f_cols, f_labels, pairs = [], {}, []
    for var, flag in variable_flags.items():
        if flag == "N": continue
        name = original_names.get(var, var)
        if "L" in flag:
            c = f"{var}_level"; feat[c] = df[var]
            f_cols.append(c); f_labels[c] = f"{name} Level"
        if "M" in flag:
            c = f"{var}_change"; feat[c] = df[var].diff()
            f_cols.append(c); f_labels[c] = f"{name} MoM Change"
        if "P" in flag:
            c = f"{var}_pct"; feat[c] = df[var].pct_change() * 100
            f_cols.append(c); f_labels[c] = f"{name} % Change"
        if "M" in flag and "P" in flag:
            pairs.append((f"{var}_change", f"{var}_pct", var))
    return feat.dropna().reset_index(drop=True), f_cols, f_labels, pairs

def run_regression(feat_df, f_cols, lookback, pairs):
    from scipy import stats
    n = len(feat_df)
    X_raw, y_raw = feat_df[f_cols].values, feat_df["hy_spread"].values
    c_idx = {c: i for i, c in enumerate(f_cols)}
    base = [c_idx[c] for c in f_cols if not any(c in p[:2] for p in pairs)]
    
    preds, coef_history, selections = np.full(n, np.nan), [], {}
    for i in range(lookback, n):
        idx, names = list(base), [f_cols[j] for j in base]
        y_train = y_raw[i-lookback:i]
        for m, p, var in pairs:
            m_v, p_v = X_raw[i-lookback:i, c_idx[m]], X_raw[i-lookback:i, c_idx[p]]
            mask = ~(np.isnan(m_v) | np.isnan(p_v) | np.isnan(y_train))
            m_c = abs(np.corrcoef(m_v[mask], y_train[mask])[0,1]) if mask.sum() > 10 else 0
            p_c = abs(np.corrcoef(p_v[mask], y_train[mask])[0,1]) if mask.sum() > 10 else 0
            sel = p if p_c > m_c else m
            idx.append(c_idx[sel]); names.append(sel)
            selections[var] = "pct" if sel == p else "mom"

        model = LinearRegression().fit(X_raw[i-lookback:i, idx], y_train)
        preds[i] = model.predict(X_raw[i:i+1, idx])[0]
        coef_history.append({"idx": i, **{f"coef_{n}": v for n, v in zip(names, model.coef_)}})

    # Final window stats with QR/Scaling
    last_idx = [c_idx[c] for c in f_cols if not any(c in p[:2] for p in pairs)]
    for m, p, var in pairs: last_idx.append(c_idx[p if selections.get(var) == "pct" else m])
    
    X_fin = X_raw[n-lookback:n, last_idx]
    y_fin = y_raw[n-lookback:n]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_fin)
    X_d = np.column_stack([np.ones(X_s.shape[0]), X_s])
    Q, R = np.linalg.qr(X_d)

    # Guard against singular R matrix (collinear or constant features)
    try:
        beta_s = np.linalg.solve(R, Q.T @ y_fin)
        R_inv = np.linalg.inv(R @ R.T)
    except np.linalg.LinAlgError:
        # Fallback: least-squares solution with pseudo-inverse
        beta_s, _, _, _ = np.linalg.lstsq(X_d, y_fin, rcond=None)
        R_inv = np.linalg.pinv(R @ R.T)

    res = y_fin - X_d @ beta_s
    dof = max(X_fin.shape[0] - X_fin.shape[1] - 1, 1)
    sigma2 = np.sum(res**2) / dof
    se_s = np.sqrt(np.abs(np.diag(sigma2 * R_inv)))
    
    model_stats = {
        "coefficients": beta_s[1:] / scaler.scale_,
        "std_errors": se_s[1:] / scaler.scale_,
        "r2": 1 - np.sum(res**2)/np.sum((y_fin - y_fin.mean())**2),
        "active_cols": [f_cols[j] for j in last_idx]
    }
    return feat_df.assign(predicted=preds, residual=feat_df["hy_spread"]-preds), model_stats, pd.DataFrame(coef_history)


# ─────────────────────────────────────────────────────────────
# Monthly Attribution (Enhanced)
# ─────────────────────────────────────────────────────────────

def monthly_attribution(results, coef_df, f_cols, f_labels):
    """
    Calculate each factor's contribution to the monthly change in
    model-predicted spread. Returns a DataFrame sorted by absolute
    impact with contributions expressed in both spread and bps terms.
    """
    if len(coef_df) < 2:
        return None

    latest, prev = coef_df.iloc[-1], coef_df.iloc[-2]
    data = []

    for c in f_cols:
        coef_val = latest.get(f"coef_{c}", 0)
        if coef_val == 0 or pd.isna(coef_val):
            continue

        latest_idx = int(latest["idx"])
        prev_idx = int(prev["idx"])
        factor_change = results[c].iloc[latest_idx] - results[c].iloc[prev_idx]
        contribution = coef_val * factor_change

        data.append({
            "Factor": f_labels.get(c, c),
            "Coefficient": coef_val,
            "Factor Change": factor_change,
            "Contribution": contribution,
            "Contribution (bps)": contribution * 100,
        })

    if not data:
        return None

    attr_df = pd.DataFrame(data)
    # Sort by absolute contribution descending
    attr_df = attr_df.sort_values(
        "Contribution (bps)", key=abs, ascending=False
    ).reset_index(drop=True)

    return attr_df


# ─────────────────────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────────────────────

def main():
    st.sidebar.title("HY Rich/Cheap Model")
    up = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
    lb = st.sidebar.select_slider("Lookback", options=[36, 48, 60], value=60)
    
    if up:
        df, dep, flags, names = load_data(up)
        if df is not None:
            feat, f_cols, f_labels, pairs = engineer_features(df, flags, names)
            res, stats, c_df = run_regression(feat, f_cols, lb, pairs)
            
            # Z-Score and Signal
            valid_res = res["residual"].dropna()
            res["z"] = (res["residual"] - valid_res.mean()) / valid_res.std()
            curr = res.iloc[-1]
            sig = "CHEAP" if curr["z"] > 1.0 else ("RICH" if curr["z"] < -2.0 else "NEUTRAL")
            
            # ── Signal Header ─────────────────────────────────
            st.markdown(
                f'<div class="signal-box signal-{sig.lower()}">'
                f'<h1>{sig}</h1>Z-Score: {curr["z"]:.2f}</div>',
                unsafe_allow_html=True,
            )
            
            # ── Commentary & Attribution ──────────────────────
            st.subheader("📝 Commentary & Attribution")
            attr = monthly_attribution(res, c_df, stats["active_cols"], f_labels)

            if attr is not None and len(attr) > 0:
                # Net move across all factors (stable even when near zero)
                net_move = attr["Contribution (bps)"].sum()
                top = attr.head(3)

                # Build narrative lines
                lines = [
                    f"**Net Model Move:** The model's fair-value spread "
                    f"moved by **{net_move:+.1f} bps** this month."
                ]

                driver_labels = [
                    "Primary Driver",
                    "Secondary Driver",
                    "Tertiary Driver",
                ]
                for i, label in enumerate(driver_labels):
                    if i < len(top):
                        row = top.iloc[i]
                        lines.append(
                            f"**{label}:** {row['Factor']} "
                            f"({row['Contribution (bps)']:+.1f} bps)"
                        )

                lines.append(
                    f"**Signal Status:** Spreads are currently **{sig}** "
                    f"with a Z-Score of **{curr['z']:.2f}** "
                    f"(R²: {stats['r2']:.2f})."
                )

                st.info("\n\n".join(lines))

                # Filtered detail table — only factors >= 0.5 bps absolute
                material = attr[attr["Contribution (bps)"].abs() >= 0.5].copy()

                with st.expander(
                    f"Detailed Factor Attribution "
                    f"({len(material)} material factors)"
                ):
                    if len(material) > 0:
                        display_df = material[
                            ["Factor", "Contribution (bps)"]
                        ].copy()
                        display_df["Contribution (bps)"] = display_df[
                            "Contribution (bps)"
                        ].map("{:+.1f}".format)
                        st.table(display_df.reset_index(drop=True))
                    else:
                        st.caption(
                            "No factors with impact ≥ 0.5 bps this month."
                        )

                # Transparency note for filtered noise
                noise_count = len(attr) - len(material)
                if noise_count > 0:
                    st.caption(
                        f"ℹ️ {noise_count} factor(s) with < 0.5 bps "
                        f"impact hidden."
                    )
            else:
                st.info(
                    "Attribution data not yet available "
                    "(requires at least 2 regression windows)."
                )

            # ── Charts ────────────────────────────────────────
            st.plotly_chart(
                go.Figure([
                    go.Scatter(
                        x=res.date, y=res.hy_spread, name="Actual"
                    ),
                    go.Scatter(
                        x=res.date, y=res.predicted, name="Model",
                        line=dict(dash="dash"),
                    ),
                ]),
                use_container_width=True,
            )

if __name__ == "__main__":
    main()
