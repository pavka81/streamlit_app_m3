import os
import math
import pandas as pd
import streamlit as st

# ---- Snowflake session: Cloud (secrets) OR inside Snowflake (active session)
session = None
try:
    # Works only inside Snowflake Streamlit/Notebook
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
except Exception:
    pass

if session is None:
    # Streamlit Cloud path: build a session from .streamlit/secrets.toml
    from snowflake.snowpark import Session
    conn = {
        "account":   st.secrets["snowflake"]["account"],
        "user":      st.secrets["snowflake"]["user"],
        "password":  st.secrets["snowflake"]["password"],
        "role":      st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database":  st.secrets["snowflake"]["database"],
        "schema":    st.secrets["snowflake"]["schema"],
    }
    session = Session.builder.configs(conn).create()

st.title("Avalanche Streamlit App (Cloud)")

# ---------------------------
# Helper: call Cortex via SQL
# ---------------------------
def cortex_complete(model: str, prompt: str) -> str:
    # escape single quotes for SQL string literal
    safe = prompt.replace("'", "''")
    sql = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{safe}') AS RESP"
    return session.sql(sql).to_pandas().iloc[0, 0]

# ---------------------------
# Load reviews (keep simple)
# ---------------------------
df_reviews = session.sql("SELECT * FROM REVIEWS_WITH_SENTIMENT").to_pandas()

# Optional: try to parse likely date columns if they exist
for c in ["REVIEW_DATE", "SHIPPING_DATE"]:
    if c in df_reviews.columns:
        df_reviews[c] = pd.to_datetime(df_reviews[c], errors="coerce")

st.caption(f"Loaded {len(df_reviews):,} rows from REVIEWS_WITH_SENTIMENT")

# ---------------------------
# Average Sentiment by Product (robust to missing columns)
# ---------------------------
st.subheader("Average Sentiment by Product")
if "PRODUCT" in df_reviews.columns and "SENTIMENT_SCORE" in df_reviews.columns:
    product_sent = (df_reviews
                    .assign(SENTIMENT_SCORE=pd.to_numeric(df_reviews["SENTIMENT_SCORE"], errors="coerce"))
                    .groupby("PRODUCT")["SENTIMENT_SCORE"]
                    .mean()
                    .sort_values())
    if len(product_sent):
        st.bar_chart(product_sent)
    else:
        st.info("No sentiment scores to aggregate.")
else:
    st.info("Need columns PRODUCT and SENTIMENT_SCORE for this chart.")

# ---------------------------
# Product filter + table
# ---------------------------
st.subheader("Filter by Product")
if "PRODUCT" in df_reviews.columns:
    product = st.selectbox("Choose a product",
                           ["All Products"] + sorted(df_reviews["PRODUCT"].dropna().unique().tolist()))
    if product != "All Products":
        filtered = df_reviews[df_reviews["PRODUCT"] == product]
    else:
        filtered = df_reviews
else:
    product = "All Products"
    filtered = df_reviews
    st.warning("PRODUCT column not found; showing all rows.")

st.subheader(f"📁 Reviews for {product}")
st.dataframe(filtered, use_container_width=True)

# ---------------------------
# Sentiment score histogram
# ---------------------------
st.subheader(f"Sentiment Score Distribution ({product})")
if "SENTIMENT_SCORE" in filtered.columns:
    s = pd.to_numeric(filtered["SENTIMENT_SCORE"], errors="coerce").dropna()
    if len(s):
        # histogram with Streamlit: bin via pandas.cut then bar
        bins = int(max(5, min(40, math.sqrt(len(s)))))
        hist = pd.cut(s, bins=bins).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)
    else:
        st.info("No numeric SENTIMENT_SCORE values to plot.")
else:
    st.info("SENTIMENT_SCORE column not found.")

# ---------------------------
# 💬 Cortex Chatbox (Cloud-safe)
# ---------------------------
st.markdown("---")
st.header("💬 Avalanche Assistant (Cortex)")

# Use a model that’s enabled in your Snowflake account/region
model = st.selectbox("Cortex model", ["mistral-large2", "snowflake-arctic", "llama3-70b"], index=0)

# small context: just column names + row count to keep prompt short
context = f"Table REVIEWS_WITH_SENTIMENT has columns: {', '.join(df_reviews.columns)}. Total rows: {len(df_reviews)}."

if "chat_msgs" not in st.session_state:
    st.session_state.chat_msgs = []

for role, msg in st.session_state.chat_msgs:
    with st.chat_message(role):
        st.write(msg)

user_q = st.chat_input("Ask a question (I can also draft Snowflake SQL)")
if user_q:
    st.session_state.chat_msgs.append(("user", user_q))
    with st.chat_message("user"):
        st.write(user_q)

    system = (
        "You are a helpful Snowflake data assistant. "
        "When asked for metrics, provide a concise Snowflake SQL the user could run. "
        "Use the table name REVIEWS_WITH_SENTIMENT and existing column names."
    )
    prompt = f"{system}\n\nContext: {context}\n\nUser: {user_q}\nAssistant:"

    try:
        reply = cortex_complete(model, prompt)
    except Exception as e:
        reply = f"Sorry, Cortex call failed: {e}"

    with st.chat_message("assistant"):
        st.write(reply)
    st.session_state.chat_msgs.append(("assistant", reply))
