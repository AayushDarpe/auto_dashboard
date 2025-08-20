import io
import textwrap
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Page & Layout
# -----------------------------
st.set_page_config(page_title="Auto Dashboard", layout="wide")
st.title("🤖 Auto Dashboard")
st.caption("Upload a dataset and get instant insights, filters, and interactive charts.")

# -----------------------------
# Helpers
# -----------------------------
COUNTRY_COL_CANDIDATES = [
    "country", "Country", "Country/other", "nation", "Nation", "country_name", "Country Name"
]
LAT_CANDIDATES = ["lat", "latitude", "Lat", "Latitude"]
LON_CANDIDATES = ["lon", "lng", "longitude", "Long", "Longitude", "Lng"]


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], str | None, str | None, str | None]:
    # Try to coerce "date-like" columns safely
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric / categorical / datetime
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    # Try to find one country col
    country_col = next((c for c in COUNTRY_COL_CANDIDATES if c in df.columns), None)

    # Find latitude/longitude columns (first match)
    lat_col = next((c for c in df.columns if any(key.lower() == c.lower() for key in LAT_CANDIDATES)), None)
    lon_col = next((c for c in df.columns if any(key.lower() == c.lower() for key in LON_CANDIDATES)), None)

    return numeric_cols, categorical_cols, datetime_cols, country_col, lat_col, lon_col


def value_counts_df(series: pd.Series, top_n: int = 50) -> pd.DataFrame:
    vc = series.value_counts(dropna=False).head(top_n).reset_index()
    # Rename to avoid "index" surprises
    vc.columns = [series.name if series.name else "Category", "count"]
    return vc


def fig_download_button(fig, label="⬇️ Download this chart (HTML)", key="dlfig"):
    if fig is None:
        return
    html_bytes = fig.to_html(full_html=False, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        label=label,
        data=html_bytes,
        file_name="chart.html",
        mime="text/html",
        key=key,
        use_container_width=True,
    )


def mk_insights(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], datetime_cols: List[str]) -> str:
    lines = []
    # Missing values
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        lines.append(f"• ⚠️ Dataset contains **{total_missing}** missing values across all columns.")
    else:
        lines.append("• ✅ No missing values detected.")

    # Numeric highlights
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        top_mean = desc["mean"].sort_values(ascending=False).head(1)
        lines.append(
            f"• 📈 Highest mean among numerics: **{top_mean.index[0]} = {top_mean.values[0]:,.2f}**."
        )
        # Simple outlier count via IQR
        outlier_report = []
        for col in numeric_cols[:7]:  # cap to keep it snappy
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if pd.isna(IQR) or IQR == 0:
                continue
            mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            n_out = int(mask.sum())
            if n_out > 0:
                outlier_report.append(f"{col} ({n_out})")
        if outlier_report:
            lines.append("• 🚨 Potential outliers detected in: " + ", ".join(outlier_report) + ".")

    # Categorical highlights
    if categorical_cols:
        col = categorical_cols[0]
        top_cat = df[col].value_counts(dropna=False).idxmax()
        lines.append(f"• 🔤 Most frequent category in **{col}**: **{top_cat}**.")

    # Date range
    if datetime_cols:
        dcol = datetime_cols[0]
        dmin, dmax = df[dcol].min(), df[dcol].max()
        if pd.notna(dmin) and pd.notna(dmax):
            lines.append(f"• ⏳ Date span in **{dcol}**: **{dmin.date()} → {dmax.date()}**.")

    # Strong correlations (abs > 0.7)
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        high_pairs = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1 :]:
                val = corr.loc[a, b]
                if pd.notna(val) and val > 0.7:
                    high_pairs.append(f"{a} ↔ {b} ({val:.2f})")
        if high_pairs:
            lines.append("• 🔗 Strong correlations: " + "; ".join(high_pairs) + ".")

    return "\n".join(lines)


def add_download_filtered(df_filtered: pd.DataFrame):
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", csv, "filtered_data.csv", "text/csv", use_container_width=True)


# -----------------------------
# File Upload
# -----------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

# Load & Detect
df = load_data(uploaded)
numeric_cols, categorical_cols, datetime_cols, country_col, lat_col, lon_col = detect_columns(df)

# -----------------------------
# Sidebar – Global Filters
# -----------------------------
st.sidebar.header("🧰 Global Filters")

# Categorical filters
if categorical_cols:
    with st.sidebar.expander("Categorical filters", expanded=False):
        for col in categorical_cols[:8]:  # limit to 8 for sanity
            choices = st.multiselect(f"{col}", sorted(df[col].dropna().unique().tolist())[:200])
            if choices:
                df = df[df[col].isin(choices)]

# Numeric range filters
if numeric_cols:
    with st.sidebar.expander("Numeric filters", expanded=False):
        for col in numeric_cols[:6]:
            col_min, col_max = float(np.nanmin(df[col])), float(np.nanmax(df[col]))
            if np.isfinite(col_min) and np.isfinite(col_max) and col_min != col_max:
                r = st.slider(f"{col} range", col_min, col_max, (col_min, col_max))
                df = df[(df[col] >= r[0]) & (df[col] <= r[1])]

# Date range filter (first datetime col)
if datetime_cols:
    dcol = datetime_cols[0]
    dmin, dmax = df[dcol].min(), df[dcol].max()
    if pd.notna(dmin) and pd.notna(dmax) and dmin != dmax:
        dr = st.sidebar.date_input("Date range", (dmin.date(), dmax.date()))
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            df = df[(df[dcol] >= pd.Timestamp(dr[0])) & (df[dcol] <= pd.Timestamp(dr[1]))]

st.sidebar.markdown("---")
add_download_filtered(df)

# -----------------------------
# Insights Summary
# -----------------------------
st.subheader("🧠 Auto Insights Summary")
insight_text = mk_insights(df, numeric_cols, categorical_cols, datetime_cols)
st.markdown(textwrap.indent(insight_text, ""))

# -----------------------------
# Auto Charts (Quick Overview)
# -----------------------------
st.subheader("⚡ Quick Overview")

cols = st.columns(3)
with cols[0]:
    if numeric_cols:
        fig = px.histogram(df, x=numeric_cols[0], nbins=30, title=f"Distribution of {numeric_cols[0]}")
        st.plotly_chart(fig, use_container_width=True)
        fig_download_button(fig, key="q1")
with cols[1]:
    if categorical_cols:
        vc = value_counts_df(df[categorical_cols[0]], top_n=10)
        fig = px.bar(vc, x=vc.columns[0], y="count", title=f"Top {vc.columns[0]}", color=vc.columns[0])
        st.plotly_chart(fig, use_container_width=True)
        fig_download_button(fig, key="q2")
with cols[2]:
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                         title=f"{numeric_cols[0]} vs {numeric_cols[1]}", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        fig_download_button(fig, key="q3")

# Time-series quick chart
if datetime_cols and numeric_cols:
    st.markdown("#### ⏳ Time Series")
    ts_metric = st.selectbox("Metric for time series", numeric_cols, key="ts_metric")
    ts_col = datetime_cols[0]
    ts_df = df[[ts_col, ts_metric]].dropna()
    if not ts_df.empty:
        ts_df = ts_df.groupby(ts_col)[ts_metric].sum().reset_index().sort_values(ts_col)
        fig = px.line(ts_df, x=ts_col, y=ts_metric, title=f"{ts_metric} over {ts_col}")
        st.plotly_chart(fig, use_container_width=True)
        fig_download_button(fig, key="tsline")

# -----------------------------
# Interactive Chart Builder
# -----------------------------
st.subheader("🛠️ Interactive Chart Builder")

chart = st.selectbox(
    "Choose a chart type",
    ["Histogram", "Bar (Top-N)", "Pie", "Line (Time)", "Scatter", "Boxplot", "Heatmap (Correlation)", "Map"],
)

built_fig = None

if chart == "Histogram":
    if numeric_cols:
        xcol = st.selectbox("Numeric column", numeric_cols, key="hist_x")
        bins = st.slider("Bins", 5, 100, 30)
        built_fig = px.histogram(df, x=xcol, nbins=bins, title=f"Histogram of {xcol}")
        st.plotly_chart(built_fig, use_container_width=True)

elif chart == "Bar (Top-N)":
    if categorical_cols:
        xcol = st.selectbox("Categorical column", categorical_cols, key="bar_cat")
        topn = st.slider("Top N", 3, 50, 10)
        vc = value_counts_df(df[xcol], top_n=topn)
        built_fig = px.bar(vc, x=vc.columns[0], y="count", title=f"Top {topn} {vc.columns[0]}", color=vc.columns[0])
        st.plotly_chart(built_fig, use_container_width=True)

elif chart == "Pie":
    if categorical_cols:
        xcol = st.selectbox("Categorical column", categorical_cols, key="pie_cat")
        vc = value_counts_df(df[xcol], top_n=25)
        # Avoid too many slices
        if len(vc) > 20:
            vc = vc.head(20)
        built_fig = px.pie(vc, names=vc.columns[0], values="count", title=f"Distribution of {vc.columns[0]}")
        st.plotly_chart(built_fig, use_container_width=True)

elif chart == "Line (Time)":
    if datetime_cols and numeric_cols:
        tcol = st.selectbox("Time column", datetime_cols, key="line_time")
        ycol = st.selectbox("Numeric metric", numeric_cols, key="line_metric")
        how = st.selectbox("Aggregation", ["sum", "mean", "median", "max", "min"], index=0)
        tmp = df[[tcol, ycol]].dropna()
        if not tmp.empty:
            g = tmp.groupby(tcol)[ycol]
            agg = getattr(g, how)().reset_index().sort_values(tcol)
            built_fig = px.line(agg, x=tcol, y=ycol, title=f"{how}({ycol}) over {tcol}")
            st.plotly_chart(built_fig, use_container_width=True)

elif chart == "Scatter":
    if len(numeric_cols) >= 2:
        xcol = st.selectbox("X (numeric)", numeric_cols, key="sc_x")
        ycol = st.selectbox("Y (numeric)", [c for c in numeric_cols if c != xcol], key="sc_y")
        color_by = st.selectbox("Color by (optional)", ["(none)"] + categorical_cols + numeric_cols, key="sc_c")
        size_by = st.selectbox("Size by (optional)", ["(none)"] + numeric_cols, key="sc_s")
        kwargs = {}
        if color_by != "(none)":
            kwargs["color"] = color_by
        if size_by != "(none)":
            kwargs["size"] = size_by
        built_fig = px.scatter(df, x=xcol, y=ycol, title=f"{ycol} vs {xcol}", **kwargs)
        st.plotly_chart(built_fig, use_container_width=True)

elif chart == "Boxplot":
    if numeric_cols:
        ycol = st.selectbox("Y (numeric)", numeric_cols, key="box_y")
        xcol = st.selectbox("Group by (categorical, optional)", ["(none)"] + categorical_cols, key="box_x")
        if xcol == "(none)":
            built_fig = px.box(df, y=ycol, title=f"Boxplot of {ycol}")
        else:
            built_fig = px.box(df, x=xcol, y=ycol, color=xcol, title=f"Boxplot of {ycol} by {xcol}")
        st.plotly_chart(built_fig, use_container_width=True)

elif chart == "Heatmap (Correlation)":
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        built_fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(built_fig, use_container_width=True)

elif chart == "Map":
    # Lat/Lon scatter mapbox OR Country choropleth
    if lat_col and lon_col:
        hover = st.selectbox("Hover name (optional)", ["(none)"] + categorical_cols, key="map_hover")
        color = st.selectbox("Color by (optional)", ["(none)"] + numeric_cols + categorical_cols, key="map_color")
        kwargs = {}
        if hover != "(none)":
            kwargs["hover_name"] = hover
        if color != "(none)":
            kwargs["color"] = color
        built_fig = px.scatter_mapbox(
            df.dropna(subset=[lat_col, lon_col]),
            lat=lat_col, lon=lon_col, zoom=1.4, height=500, **kwargs
        )
        built_fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(built_fig, use_container_width=True)
    elif country_col:
        # Aggregate rows per country
        agg_col = st.selectbox("Value to color by (optional)", ["(count)"] + numeric_cols, key="choropleth_val")
        if agg_col == "(count)":
            tmp = df.groupby(country_col).size().reset_index(name="value")
        else:
            how = st.selectbox("Aggregation", ["sum", "mean", "median", "max", "min"], key="choropleth_agg")
            tmp = getattr(df.groupby(country_col)[agg_col], how)().reset_index(name="value")
        built_fig = px.choropleth(
            tmp,
            locations=country_col,
            locationmode="country names",
            color="value",
            color_continuous_scale="Viridis",
            title="🌍 Choropleth by Country",
        )
        st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("No latitude/longitude or country column detected.")

# Download last built chart
if built_fig is not None:
    fig_download_button(built_fig, key="built_fig")

# Footer
st.markdown("---")
st.caption("Tip: Use the sidebar to filter your data globally. Charts respect the filtered view.")

