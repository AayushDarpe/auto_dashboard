import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Auto Dashboard", layout="wide")
st.title("🤖 Smart Auto-Dashboard Generator")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset uploaded successfully!")

    # Preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Info
    st.subheader("📑 Dataset Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.write("Columns:", list(df.columns))

    # Auto-convert datetimes
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            pass

    # Detect column types
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    st.subheader("📊 Auto-Generated Dashboard")

    # --- Numeric Column Charts ---
    if numeric_cols:
        st.markdown("### 📈 Numeric Data Insights")
        for col in numeric_cols[:2]:  # limit to 2 charts
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Categorical Column Charts ---
    if categorical_cols:
        st.markdown("### 🏷️ Categorical Data Insights")
        for col in categorical_cols[:2]:  # limit to 2 charts
            if df[col].nunique() > 50:  # skip if too many categories
                continue
            top_counts = df[col].value_counts().nlargest(10).reset_index()
            top_counts.columns = [col, "count"]  # Fix column names
            fig = px.bar(top_counts, x=col, y="count", title=f"Top 10 {col}")
            st.plotly_chart(fig, use_container_width=True)

        # Auto Pie Chart
        if df[categorical_cols[0]].nunique() <= 20:  # avoid too many slices
            fig = px.pie(df, names=categorical_cols[0], title=f"Distribution of {categorical_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Time Series ---
    if datetime_cols and numeric_cols:
        st.markdown("### ⏳ Time-Series Insights")
        fig = px.line(df, x=datetime_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} over {datetime_cols[0]}")
        st.plotly_chart(fig, use_container_width=True)

    # --- Scatter Plot (Top 2 numeric columns) ---
    if len(numeric_cols) >= 2:
        st.markdown("### 🔗 Relationship Insights")
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
        st.plotly_chart(fig, use_container_width=True)

    # --- Map Visualization ---
    if lat_cols and lon_cols:
        st.markdown("### 🌍 Geographical Insights")
        fig = px.scatter_mapbox(
            df, lat=lat_cols[0], lon=lon_cols[0],
            hover_name=categorical_cols[0] if categorical_cols else None,
            color=numeric_cols[0] if numeric_cols else None,
            zoom=2, height=400
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
