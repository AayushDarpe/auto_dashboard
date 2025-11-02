import io
import textwrap
from typing import List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------
# Page & Layout
# -----------------------------
st.set_page_config(
    page_title="Auto Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Auto Dashboard - Instant insights from your data"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Auto Dashboard")
st.caption("Upload a dataset and get instant insights, filters, and interactive charts • Powered by Streamlit")

# -----------------------------
# Constants & Helpers
# -----------------------------
COUNTRY_COL_CANDIDATES = [
    "country", "Country", "Country/other", "nation", "Nation", 
    "country_name", "Country Name", "countries", "Countries",
    "country_code", "Country Code", "iso", "ISO"
]
LAT_CANDIDATES = ["lat", "latitude", "Lat", "Latitude", "LAT", "y", "Y"]
LON_CANDIDATES = ["lon", "lng", "longitude", "Long", "Longitude", "Lng", "LON", "x", "X"]

MAX_CATEGORICAL_FILTERS = 10
MAX_NUMERIC_FILTERS = 8
MAX_UNIQUE_FOR_CATEGORICAL = 1000
MAX_ROWS_FOR_DETAILED_ANALYSIS = 100000


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """Load CSV or Excel file with better error handling."""
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            # Try different encodings
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin-1')
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, engine='openpyxl' if name.endswith('.xlsx') else None)
        else:
            st.error(f"Unsupported file format: {name}")
            return pd.DataFrame()
        
        # Basic validation
        if df.empty:
            st.warning("The uploaded file is empty.")
            return df
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], Optional[str], Optional[str], Optional[str]]:
    """Detect column types and special columns (country, lat, lon)."""
    if df.empty:
        return [], [], [], None, None, None
    
    # Try to coerce date-like columns safely
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["date", "time", "datetime", "timestamp"]):
            try:
                # Check if column looks like it could be a date
                sample = df[col].dropna().head(100)
                if len(sample) > 0 and not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except:
                pass

    # Numeric / categorical / datetime
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    
    # Categorical: object/category types with reasonable cardinality
    categorical_cols = []
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        unique_count = df[col].nunique()
        if 1 < unique_count < MAX_UNIQUE_FOR_CATEGORICAL:  # Skip constant columns
            categorical_cols.append(col)

    # Try to find country column
    country_col = next((c for c in COUNTRY_COL_CANDIDATES if c in df.columns), None)

    # Find latitude/longitude columns (case-insensitive)
    lat_col = next((c for c in df.columns if any(key.lower() == c.lower() for key in LAT_CANDIDATES)), None)
    lon_col = next((c for c in df.columns if any(key.lower() == c.lower() for key in LON_CANDIDATES)), None)

    return numeric_cols, categorical_cols, datetime_cols, country_col, lat_col, lon_col


def detect_data_quality_issues(df: pd.DataFrame) -> List[str]:
    """Detect potential data quality issues."""
    issues = []
    
    # Check for duplicate columns
    if len(df.columns) != len(set(df.columns)):
        issues.append("⚠️ Duplicate column names detected")
    
    # Check for columns with single value
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"⚠️ Column '{col}' has only one unique value")
    
    # Check for high percentage of missing values
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        if missing_pct > 50:
            issues.append(f"⚠️ Column '{col}' is {missing_pct:.1f}% missing")
    
    return issues


def value_counts_df(series: pd.Series, top_n: int = 50) -> pd.DataFrame:
    """Create a dataframe from value counts."""
    vc = series.value_counts(dropna=False).head(top_n).reset_index()
    vc.columns = [series.name if series.name else "Category", "count"]
    return vc


def fig_download_button(fig, label="⬇️ Download Chart (HTML)", key="dlfig"):
    """Create a download button for a Plotly figure."""
    if fig is None:
        return
    try:
        html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label=label,
            data=html_bytes,
            file_name=f"chart_{key}.html",
            mime="text/html",
            key=key,
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")


def create_summary_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Create enhanced summary statistics."""
    if not numeric_cols:
        return pd.DataFrame()
    
    stats = df[numeric_cols].describe().T
    stats['missing'] = df[numeric_cols].isna().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    stats['zeros'] = (df[numeric_cols] == 0).sum()
    stats['unique'] = df[numeric_cols].nunique()
    
    return stats


def mk_insights(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], datetime_cols: List[str]) -> str:
    """Generate automatic insights from the dataset."""
    lines = []
    
    # Basic stats
    lines.append(f"• 📊 **{len(df):,}** rows × **{len(df.columns)}** columns")
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    lines.append(f"• 💾 Memory usage: **{memory_usage:.2f} MB**")
    
    # Missing values
    total_missing = int(df.isna().sum().sum())
    total_cells = len(df) * len(df.columns)
    missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    if total_missing > 0:
        lines.append(f"• ⚠️ **{total_missing:,}** missing values ({missing_pct:.1f}% of all cells)")
        # Show columns with most missing data
        missing_by_col = df.isna().sum().sort_values(ascending=False).head(3)
        missing_by_col = missing_by_col[missing_by_col > 0]
        if len(missing_by_col) > 0:
            top_missing = ", ".join([f"{col} ({count})" for col, count in missing_by_col.items()])
            lines.append(f"  - Most affected columns: {top_missing}")
    else:
        lines.append("• ✅ No missing values detected")

    # Numeric highlights
    if numeric_cols:
        try:
            desc = df[numeric_cols].describe().T
            if not desc.empty and 'mean' in desc.columns:
                # Highest mean
                top_mean = desc["mean"].sort_values(ascending=False).head(1)
                if len(top_mean) > 0:
                    lines.append(f"• 📈 Highest mean: **{top_mean.index[0]}** = **{top_mean.values[0]:,.2f}**")
                
                # Most variable (highest std)
                if 'std' in desc.columns:
                    top_std = desc["std"].sort_values(ascending=False).head(1)
                    if len(top_std) > 0 and pd.notna(top_std.values[0]):
                        lines.append(f"• 📊 Most variable: **{top_std.index[0]}** (σ = {top_std.values[0]:,.2f})")
            
            # Outlier detection (limited to avoid performance issues)
            outlier_report = []
            for col in numeric_cols[:7]:
                try:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    if pd.notna(IQR) and IQR > 0:
                        mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                        n_out = int(mask.sum())
                        if n_out > 0:
                            pct = (n_out / len(df)) * 100
                            outlier_report.append(f"{col} ({n_out}, {pct:.1f}%)")
                except:
                    continue
            
            if outlier_report:
                lines.append("• 🚨 Potential outliers: " + ", ".join(outlier_report[:3]))
        except:
            pass

    # Categorical highlights
    if categorical_cols:
        try:
            col = categorical_cols[0]
            vc = df[col].value_counts(dropna=False)
            if len(vc) > 0:
                top_cat = vc.idxmax()
                top_count = vc.max()
                pct = (top_count / len(df)) * 100 if len(df) > 0 else 0
                unique_count = df[col].nunique()
                lines.append(f"• 🔤 **{col}**: {unique_count:,} unique values, most frequent is **{top_cat}** ({top_count:,}, {pct:.1f}%)")
        except:
            pass

    # Date range
    if datetime_cols:
        try:
            dcol = datetime_cols[0]
            valid_dates = df[dcol].dropna()
            if len(valid_dates) > 0:
                dmin, dmax = valid_dates.min(), valid_dates.max()
                if pd.notna(dmin) and pd.notna(dmax):
                    days = (dmax - dmin).days
                    lines.append(f"• ⏳ **{dcol}**: {dmin.date()} → {dmax.date()} ({days:,} days span)")
        except:
            pass

    # Strong correlations
    if len(numeric_cols) > 1:
        try:
            corr = df[numeric_cols].corr().abs()
            high_pairs = []
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i + 1:]:
                    val = corr.loc[a, b]
                    if pd.notna(val) and val > 0.7:
                        high_pairs.append(f"{a} ↔ {b} ({val:.2f})")
                        if len(high_pairs) >= 3:
                            break
                if len(high_pairs) >= 3:
                    break
            if high_pairs:
                lines.append("• 🔗 Strong correlations: " + "; ".join(high_pairs))
        except:
            pass

    # Duplicate rows
    try:
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            pct = (n_dupes / len(df)) * 100
            lines.append(f"• 🔄 **{n_dupes:,}** duplicate rows ({pct:.1f}%)")
    except:
        pass

    return "\n".join(lines)


def add_download_filtered(df_filtered: pd.DataFrame):
    """Add download buttons for filtered data in multiple formats."""
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV", 
                csv, 
                "filtered_data.csv", 
                "text/csv", 
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating CSV download: {str(e)}")
    
    with col2:
        try:
            # Excel download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_filtered.to_excel(writer, index=False, sheet_name='Data')
            excel_data = buffer.getvalue()
            st.download_button(
                "⬇️ Download Excel",
                excel_data,
                "filtered_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating Excel download: {str(e)}")


def safe_plot(plot_func, error_msg="Error creating chart"):
    """Wrapper for safe plotting with error handling."""
    try:
        return plot_func()
    except Exception as e:
        st.error(f"{error_msg}: {str(e)}")
        return None


def export_report(df: pd.DataFrame, insights: str, charts_config: dict):
    """Generate a downloadable report with insights and metadata."""
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
        },
        "insights": insights,
        "summary_statistics": df.describe().to_dict() if not df.empty else {}
    }
    
    json_str = json.dumps(report, indent=2, default=str)
    st.download_button(
        "📄 Download Full Report (JSON)",
        json_str.encode('utf-8'),
        "dashboard_report.json",
        "application/json",
        use_container_width=True
    )


# -----------------------------
# File Upload
# -----------------------------
uploaded = st.file_uploader(
    "📁 Upload your dataset", 
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel (.xlsx, .xls). Max file size: 200MB"
)

if not uploaded:
    st.info("👋 Welcome! Upload a dataset to begin exploring your data")
    
    # Create example/demo section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Features
        - 🔍 **Automatic data type detection**
        - 📊 **Interactive charts and visualizations**
        - 🎯 **Dynamic filtering**
        - 🗺️ **Geospatial mapping** (with lat/lon or country data)
        - 📈 **Time series analysis**
        - 💾 **Export filtered data and charts**
        - 🧠 **AI-powered insights**
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Supported Chart Types
        - Histogram, Bar, Pie charts
        - Line charts (time series)
        - Scatter plots with trend lines
        - Box plots & Violin plots
        - Correlation heatmaps
        - Geographic maps (choropleth & scatter)
        """)
    
    st.markdown("---")
    st.markdown("💡 **Quick Start:** Upload a CSV or Excel file to automatically generate insights and visualizations")
    st.stop()

# Load & Detect
with st.spinner("🔄 Loading and analyzing your data..."):
    df = load_data(uploaded)

if df.empty:
    st.error("❌ Failed to load data or file is empty. Please check your file and try again.")
    st.stop()

# Store original for reset functionality
if 'original_df' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded.name:
    st.session_state.original_df = df.copy()
    st.session_state.uploaded_file_name = uploaded.name
    st.session_state.filter_applied = False

numeric_cols, categorical_cols, datetime_cols, country_col, lat_col, lon_col = detect_columns(df)

# Check for data quality issues
quality_issues = detect_data_quality_issues(df)

# -----------------------------
# Sidebar – Global Filters
# -----------------------------
st.sidebar.header("🧰 Global Filters")

# Reset button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True, type="primary"):
    df = st.session_state.original_df.copy()
    st.session_state.filter_applied = False
    st.rerun()

# Quick stats in sidebar
with st.sidebar.expander("📊 Quick Stats", expanded=True):
    st.metric("Total Rows", f"{len(df):,}")
    st.metric("Total Columns", len(df.columns))
    if numeric_cols:
        st.metric("Numeric Columns", len(numeric_cols))
    if categorical_cols:
        st.metric("Categorical Columns", len(categorical_cols))
    if datetime_cols:
        st.metric("Date Columns", len(datetime_cols))

# Categorical filters
if categorical_cols:
    with st.sidebar.expander("📋 Categorical Filters", expanded=False):
        for col in categorical_cols[:MAX_CATEGORICAL_FILTERS]:
            unique_vals = sorted([str(x) for x in df[col].dropna().unique()])[:500]
            if unique_vals:
                choices = st.multiselect(
                    f"{col}",
                    unique_vals,
                    key=f"cat_filter_{col}",
                    help=f"{len(unique_vals)} unique values"
                )
                if choices:
                    df = df[df[col].astype(str).isin(choices)]
                    st.session_state.filter_applied = True

# Numeric range filters
if numeric_cols:
    with st.sidebar.expander("🔢 Numeric Filters", expanded=False):
        for col in numeric_cols[:MAX_NUMERIC_FILTERS]:
            try:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                col_min, col_max = float(col_data.min()), float(col_data.max())
                
                if np.isfinite(col_min) and np.isfinite(col_max) and col_min != col_max:
                    r = st.slider(
                        f"{col}",
                        col_min,
                        col_max,
                        (col_min, col_max),
                        key=f"num_filter_{col}",
                        help=f"Range: {col_min:.2f} to {col_max:.2f}"
                    )
                    if r[0] != col_min or r[1] != col_max:
                        df = df[(df[col] >= r[0]) & (df[col] <= r[1])]
                        st.session_state.filter_applied = True
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not create filter for {col}")

# Date range filter
if datetime_cols:
    with st.sidebar.expander("📅 Date Filters", expanded=False):
        for dcol in datetime_cols[:3]:
            try:
                valid_dates = df[dcol].dropna()
                if len(valid_dates) == 0:
                    continue
                
                dmin, dmax = valid_dates.min(), valid_dates.max()
                
                if pd.notna(dmin) and pd.notna(dmax) and dmin != dmax:
                    dr = st.date_input(
                        f"{dcol}",
                        (dmin.date(), dmax.date()),
                        key=f"date_filter_{dcol}",
                        help=f"Range: {dmin.date()} to {dmax.date()}"
                    )
                    if isinstance(dr, (list, tuple)) and len(dr) == 2:
                        start_date, end_date = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
                        if start_date != dmin or end_date != dmax:
                            df = df[(df[dcol] >= start_date) & (df[dcol] <= end_date)]
                            st.session_state.filter_applied = True
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not create date filter for {dcol}")

st.sidebar.markdown("---")

# Filter status indicator
if st.session_state.get('filter_applied', False):
    st.sidebar.success(f"✅ Filters Active")
    
original_count = len(st.session_state.original_df)
filtered_count = len(df)
filter_pct = (filtered_count / original_count * 100) if original_count > 0 else 0

st.sidebar.markdown(f"**Showing:** {filtered_count:,} / {original_count:,} rows ({filter_pct:.1f}%)")

# Download section
st.sidebar.markdown("### 💾 Export Data")
add_download_filtered(df)

# Check if data was filtered out completely
if df.empty:
    st.warning("⚠️ All data has been filtered out. Please adjust your filters.")
    if st.button("🔄 Reset Filters"):
        df = st.session_state.original_df.copy()
        st.session_state.filter_applied = False
        st.rerun()
    st.stop()

# -----------------------------
# Data Quality Warnings
# -----------------------------
if quality_issues:
    with st.expander("⚠️ Data Quality Issues Detected", expanded=False):
        for issue in quality_issues[:5]:  # Limit to top 5
            st.warning(issue)

# -----------------------------
# Insights Summary
# -----------------------------
st.subheader("🧠 Auto Insights Summary")
with st.spinner("🔍 Analyzing data patterns..."):
    insight_text = mk_insights(df, numeric_cols, categorical_cols, datetime_cols)

st.markdown(insight_text)

# Add export report button
col1, col2, col3 = st.columns([2, 1, 1])
with col3:
    export_report(df, insight_text, {})

# -----------------------------
# Summary Statistics Table
# -----------------------------
if numeric_cols:
    with st.expander("📊 Detailed Summary Statistics", expanded=False):
        summary_stats = create_summary_stats(df, numeric_cols)
        st.dataframe(
            summary_stats.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                'max': '{:.2f}',
                'missing_pct': '{:.1f}%'
            }),
            use_container_width=True
        )

# -----------------------------
# Auto Charts (Quick Overview)
# -----------------------------
st.subheader("⚡ Quick Overview")

cols = st.columns(3)

with cols[0]:
    if numeric_cols:
        def plot():
            fig = px.histogram(
                df, 
                x=numeric_cols[0], 
                nbins=30, 
                title=f"Distribution: {numeric_cols[0]}",
                color_discrete_sequence=["#636EFA"],
                labels={numeric_cols[0]: numeric_cols[0].title()}
            )
            fig.update_layout(showlegend=False, hovermode='x unified')
            return fig
        
        fig = safe_plot(plot)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            fig_download_button(fig, key="q1")
    else:
        st.info("No numeric columns available")

with cols[1]:
    if categorical_cols:
        def plot():
            vc = value_counts_df(df[categorical_cols[0]], top_n=10)
            fig = px.bar(
                vc, 
                x=vc.columns[0], 
                y="count", 
                title=f"Top 10: {vc.columns[0]}", 
                color=vc.columns[0],
                labels={'count': 'Count', vc.columns[0]: vc.columns[0].title()}
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            return fig
        
        fig = safe_plot(plot)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            fig_download_button(fig, key="q2")
    else:
        st.info("No categorical columns available")

with cols[2]:
    if len(numeric_cols) >= 2:
        def plot():
            sample_size = min(len(df), 1000)
            df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            fig = px.scatter(
                df_sample, 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                trendline="ols",
                opacity=0.6,
                labels={
                    numeric_cols[0]: numeric_cols[0].title(),
                    numeric_cols[1]: numeric_cols[1].title()
                }
            )
            if len(df) > sample_size:
                fig.add_annotation(
                    text=f"Showing {sample_size:,} of {len(df):,} points",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            return fig
        
        fig = safe_plot(plot)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            fig_download_button(fig, key="q3")
    else:
        st.info("Need 2+ numeric columns")

# Time-series quick chart
if datetime_cols and numeric_cols:
    st.markdown("#### ⏳ Time Series Preview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ts_metric = st.selectbox("Metric", numeric_cols, key="ts_metric")
    with col2:
        ts_agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count"], key="ts_agg")
    
    ts_col = datetime_cols[0]
    
    def plot_timeseries():
        ts_df = df[[ts_col, ts_metric]].dropna()
        if ts_df.empty:
            st.info("No data available for time series")
            return None
        
        if ts_agg == "count":
            ts_df = ts_df.groupby(ts_col).size().reset_index(name=ts_metric)
        else:
            ts_df = ts_df.groupby(ts_col)[ts_metric].agg(ts_agg).reset_index()
        
        ts_df = ts_df.sort_values(ts_col)
        
        fig = px.line(
            ts_df, 
            x=ts_col, 
            y=ts_metric, 
            title=f"{ts_agg.title()} of {ts_metric} over time",
            markers=True,
            labels={ts_col: "Date", ts_metric: ts_metric.title()}
        )
        fig.update_traces(line_color='#EF553B')
        return fig
    
    fig = safe_plot(plot_timeseries)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        fig_download_button(fig, key="tsline")

# -----------------------------
# Interactive Chart Builder
# -----------------------------
st.subheader("🛠️ Interactive Chart Builder")

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    chart = st.selectbox(
        "Choose a chart type",
        [
            "Histogram", 
            "Bar (Top-N)", 
            "Pie", 
            "Line (Time)", 
            "Scatter", 
            "Boxplot", 
            "Violin Plot",
            "Heatmap (Correlation)", 
            "Map"
        ],
        help="Select the type of visualization you want to create"
    )

with chart_col2:
    show_stats = st.checkbox("Show statistics", value=False, help="Display statistical information on the chart")

built_fig = None

if chart == "Histogram":
    if numeric_cols:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            xcol = st.selectbox("Numeric column", numeric_cols, key="hist_x")
        with col2:
            bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
        with col3:
            log_scale = st.checkbox("Log scale", key="hist_log")
        
        def plot():
            fig = px.histogram(
                df, 
                x=xcol, 
                nbins=bins, 
                title=f"Distribution of {xcol}",
                log_y=log_scale,
                labels={xcol: xcol.title()}
            )
            if show_stats:
                mean_val = df[xcol].mean()
                median_val = df[xcol].median()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: {mean_val:.2f}")
                fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                             annotation_text=f"Median: {median_val:.2f}")
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("No numeric columns available")

elif chart == "Bar (Top-N)":
    if categorical_cols:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            xcol = st.selectbox("Categorical column", categorical_cols, key="bar_cat")
        with col2:
            topn = st.slider("Top N", 3, 50, 10, key="bar_topn")
        with col3:
            sort_order = st.selectbox("Sort", ["Descending", "Ascending"], key="bar_sort")
        
        def plot():
            vc = value_counts_df(df[xcol], top_n=topn)
            if sort_order == "Ascending":
                vc = vc.sort_values('count')
            fig = px.bar(
                vc, 
                x=vc.columns[0], 
                y="count", 
                title=f"Top {topn} {vc.columns[0]}", 
                color=vc.columns[0],
                labels={'count': 'Count', vc.columns[0]: vc.columns[0].title()}
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("No categorical columns available")

elif chart == "Pie":
    if categorical_cols:
        col1, col2 = st.columns([2, 1])
        with col1:
            xcol = st.selectbox("Categorical column", categorical_cols, key="pie_cat")
        with col2:
            show_pct = st.checkbox("Show percentages", value=True, key="pie_pct")
        
        def plot():
            vc = value_counts_df(df[xcol], top_n=15)
            fig = px.pie(
                vc, 
                names=vc.columns[0], 
                values="count", 
                title=f"Distribution of {vc.columns[0]}",
                hole=0.3  # Donut chart
            )
            if show_pct:
                fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("No categorical columns available")

elif chart == "Line (Time)":
    if datetime_cols and numeric_cols:
        col1, col2, col3 = st.columns(3)
        with col1:
            tcol = st.selectbox("Time column", datetime_cols, key="line_time")
        with col2:
            ycol = st.selectbox("Numeric metric", numeric_cols, key="line_metric")
        with col3:
            how = st.selectbox("Aggregation", ["sum", "mean", "median", "max", "min", "count"], index=0, key="line_agg")
        
        def plot():
            tmp = df[[tcol, ycol]].dropna()
            if tmp.empty:
                st.info("No data available")
                return None
            
            if how == "count":
                agg = tmp.groupby(tcol).size().reset_index(name=ycol)
            else:
                g = tmp.groupby(tcol)[ycol]
                agg = getattr(g, how)().reset_index()
            
            agg = agg.sort_values(tcol)
            fig = px.line(
                agg, 
                x=tcol, 
                y=ycol, 
                title=f"{how.title()} of {ycol} over time", 
                markers=True,
                labels={tcol: "Date", ycol: ycol.title()}
            )
            
            if show_stats and how != "count":
                mean_val = agg[ycol].mean()
                fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                             annotation_text=f"Average: {mean_val:.2f}")
            
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("Need both datetime and numeric columns")

elif chart == "Scatter":
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            xcol = st.selectbox("X axis (numeric)", numeric_cols, key="sc_x")
            ycol = st.selectbox("Y axis (numeric)", [c for c in numeric_cols if c != xcol], key="sc_y")
        with col2:
            color_by = st.selectbox("Color by (optional)", ["(none)"] + categorical_cols + numeric_cols, key="sc_c")
            size_by = st.selectbox("Size by (optional)", ["(none)"] + numeric_cols, key="sc_s")
        
        show_trendline = st.checkbox("Show trendline", value=True, key="sc_trend")
        
        def plot():
            kwargs = {}
            if color_by != "(none)":
                kwargs["color"] = color_by
            if size_by != "(none)":
                kwargs["size"] = size_by
            if show_trendline:
                kwargs["trendline"] = "ols"
            
            plot_df = df.sample(n=min(len(df), 5000), random_state=42) if len(df) > 5000 else df
            
            fig = px.scatter(
                plot_df, 
                x=xcol, 
                y=ycol, 
                title=f"{ycol} vs {xcol}", 
                opacity=0.6,
                labels={xcol: xcol.title(), ycol: ycol.title()},
                **kwargs
            )
            
            if len(df) > 5000:
                fig.add_annotation(
                    text=f"Showing 5,000 of {len(df):,} points",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
            
            # Show correlation coefficient
            if show_stats:
                try:
                    corr = df[[xcol, ycol]].corr().iloc[0, 1]
                    st.info(f"📊 Correlation coefficient: **{corr:.3f}**")
                except:
                    pass
    else:
        st.info("Need at least 2 numeric columns")

elif chart == "Boxplot":
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            ycol = st.selectbox("Y axis (numeric)", numeric_cols, key="box_y")
        with col2:
            xcol = st.selectbox("Group by (categorical, optional)", ["(none)"] + categorical_cols, key="box_x")
        
        show_points = st.checkbox("Show all points", value=False, key="box_points")
        
        def plot():
            if xcol == "(none)":
                fig = px.box(df, y=ycol, title=f"Boxplot of {ycol}",
                           labels={ycol: ycol.title()})
            else:
                fig = px.box(df, x=xcol, y=ycol, color=xcol, 
                           title=f"Boxplot of {ycol} by {xcol}",
                           labels={xcol: xcol.title(), ycol: ycol.title()})
            
            if show_points:
                fig.update_traces(boxpoints='all', jitter=0.3, pointpos=-1.8)
            
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("No numeric columns available")

elif chart == "Violin Plot":
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            ycol = st.selectbox("Y axis (numeric)", numeric_cols, key="violin_y")
        with col2:
            xcol = st.selectbox("Group by (categorical, optional)", ["(none)"] + categorical_cols, key="violin_x")
        
        show_box = st.checkbox("Show box plot inside", value=True, key="violin_box")
        
        def plot():
            if xcol == "(none)":
                fig = px.violin(df, y=ycol, title=f"Violin Plot of {ycol}", 
                              box=show_box, labels={ycol: ycol.title()})
            else:
                fig = px.violin(df, x=xcol, y=ycol, color=xcol, 
                              title=f"Violin Plot of {ycol} by {xcol}", 
                              box=show_box,
                              labels={xcol: xcol.title(), ycol: ycol.title()})
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("No numeric columns available")

elif chart == "Heatmap (Correlation)":
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_cols = st.multiselect(
                "Select columns for correlation (leave empty for all)",
                numeric_cols,
                default=[],
                key="heatmap_cols"
            )
        with col2:
            corr_method = st.selectbox("Method", ["pearson", "spearman"], key="corr_method")
        
        def plot():
            cols_to_use = selected_cols if selected_cols else numeric_cols
            corr = df[cols_to_use].corr(method=corr_method)
            
            fig = px.imshow(
                corr, 
                text_auto=".2f", 
                aspect="auto", 
                title=f"Correlation Heatmap ({corr_method.title()})",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels=dict(color="Correlation")
            )
            fig.update_xaxes(side="bottom")
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
            
            # Show strongest correlations
            if show_stats:
                try:
                    cols_to_use = selected_cols if selected_cols else numeric_cols
                    corr = df[cols_to_use].corr(method=corr_method).abs()
                    
                    # Get top 5 correlations
                    corr_pairs = []
                    for i, a in enumerate(cols_to_use):
                        for b in cols_to_use[i+1:]:
                            val = corr.loc[a, b]
                            if pd.notna(val):
                                corr_pairs.append((a, b, val))
                    
                    corr_pairs.sort(key=lambda x: x[2], reverse=True)
                    
                    if corr_pairs:
                        st.markdown("**🔗 Strongest Correlations:**")
                        for a, b, val in corr_pairs[:5]:
                            st.write(f"• {a} ↔ {b}: {val:.3f}")
                except:
                    pass
    else:
        st.info("Need at least 2 numeric columns")

elif chart == "Map":
    if lat_col and lon_col:
        st.markdown("**Scatter Map (Latitude/Longitude)**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hover = st.selectbox("Hover name (optional)", ["(none)"] + categorical_cols, key="map_hover")
        with col2:
            color = st.selectbox("Color by (optional)", ["(none)"] + numeric_cols + categorical_cols, key="map_color")
        with col3:
            size = st.selectbox("Size by (optional)", ["(none)"] + numeric_cols, key="map_size")
        
        def plot():
            kwargs = {}
            if hover != "(none)":
                kwargs["hover_name"] = hover
            if color != "(none)":
                kwargs["color"] = color
            if size != "(none)":
                kwargs["size"] = size
            
            map_df = df.dropna(subset=[lat_col, lon_col])
            if map_df.empty:
                st.info("No valid lat/lon data")
                return None
            
            # Sample for performance
            if len(map_df) > 10000:
                map_df = map_df.sample(n=10000, random_state=42)
                st.info(f"📍 Showing 10,000 of {len(df):,} points for performance")
            
            fig = px.scatter_mapbox(
                map_df,
                lat=lat_col, 
                lon=lon_col, 
                zoom=1.4, 
                height=600,
                title=f"Geographic Distribution",
                **kwargs
            )
            fig.update_layout(
                mapbox_style="open-street-map", 
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
            
    elif country_col:
        st.markdown("**Choropleth Map (Country)**")
        
        col1, col2 = st.columns(2)
        with col1:
            agg_col = st.selectbox("Value to color by", ["(count)"] + numeric_cols, key="choropleth_val")
        with col2:
            if agg_col != "(count)":
                how = st.selectbox("Aggregation", ["sum", "mean", "median", "max", "min"], key="choropleth_agg")
        
        def plot():
            if agg_col == "(count)":
                tmp = df.groupby(country_col).size().reset_index(name="value")
                title_suffix = "by Count"
            else:
                tmp = getattr(df.groupby(country_col)[agg_col], how)().reset_index(name="value")
                title_suffix = f"by {how.title()} of {agg_col}"
            
            fig = px.choropleth(
                tmp,
                locations=country_col,
                locationmode="country names",
                color="value",
                color_continuous_scale="Viridis",
                title=f"🌍 Geographic Distribution {title_suffix}",
                height=600,
                labels={"value": title_suffix}
            )
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            return fig
        
        built_fig = safe_plot(plot)
        if built_fig:
            st.plotly_chart(built_fig, use_container_width=True)
    else:
        st.info("📍 No geographic data detected. Please ensure your data has:\n- Columns named 'lat/latitude' and 'lon/longitude', OR\n- A 'country' column with country names")

# Download last built chart
if built_fig is not None:
    fig_download_button(built_fig, key="built_fig")

# -----------------------------
# Data Preview & Search
# -----------------------------
st.subheader("📋 Data Preview & Exploration")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    show_rows = st.slider("Rows to display", 5, min(500, len(df)), min(20, len(df)), key="preview_rows")
with col2:
    search_col = st.selectbox("Search in column", ["(all)"] + df.columns.tolist(), key="search_col")
with col3:
    if search_col != "(all)":
        search_term = st.text_input("Search term", key="search_term")

# Apply search filter
display_df = df.copy()
if search_col != "(all)" and 'search_term' in st.session_state and st.session_state.search_term:
    try:
        display_df = display_df[
            display_df[search_col].astype(str).str.contains(
                st.session_state.search_term, 
                case=False, 
                na=False
            )
        ]
        st.info(f"🔍 Found {len(display_df)} matching rows")
    except:
        pass

# Show dataframe with styling
st.dataframe(
    display_df.head(show_rows),
    use_container_width=True,
    height=400
)

# Show column info
with st.expander("ℹ️ Column Information", expanded=False):
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isna().sum(),
        'Unique Values': df.nunique(),
        'Memory Usage': [f"{df[col].memory_usage(deep=True) / 1024:.1f} KB" for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💡 **Tip:** Use the sidebar to filter your data globally")
with col2:
    st.caption("📊 All charts respect the filtered view")
with col3:
    st.caption("🔄 Press 'Reset All Filters' to start fresh")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Plotly | Auto Dashboard v2.0")
