"""
Simple All-in-One Streamlit Dashboard
Everything in ONE file - no separate components
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "aerostream")
DB_USER = os.getenv("POSTGRES_USER", "aerostream_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "aerostream_pass")

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

SENTIMENT_COLORS = {
    'negative': '#EF553B',  # Red
    'neutral': '#636EFA',   # Blue
    'positive': '#00CC96'   # Green
}

SENTIMENT_LABELS = {
    'negative': 'Negative 😞',
    'neutral': 'Neutral 😐',
    'positive': 'Positive 😊'
}

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================
@st.cache_resource
def get_database_connection():
    """Create database engine (cached)"""
    return create_engine(CONNECTION_STRING, pool_pre_ping=True)

@st.cache_data(ttl=60)
def load_data_from_db(query):
    """Execute SQL query and return DataFrame"""
    try:
        engine = get_database_connection()
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

# ============================================================================
# CHART FUNCTIONS
# ============================================================================
def create_sentiment_pie_chart(df):
    """Pie chart for sentiment distribution"""
    df['label'] = df['sentiment'].map(SENTIMENT_LABELS)
    
    fig = px.pie(
        df,
        values='count',
        names='label',
        color='label',
        color_discrete_map={
            SENTIMENT_LABELS['negative']: SENTIMENT_COLORS['negative'],
            SENTIMENT_LABELS['neutral']: SENTIMENT_COLORS['neutral'],
            SENTIMENT_LABELS['positive']: SENTIMENT_COLORS['positive']
        },
        hole=0.4,
        title="Sentiment Distribution"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_airline_bar_chart(df):
    """Bar chart for sentiment by airline"""
    df['label'] = df['sentiment'].map(SENTIMENT_LABELS)
    
    fig = px.bar(
        df,
        x='airline',
        y='count',
        color='label',
        barmode='group',
        color_discrete_map={
            SENTIMENT_LABELS['negative']: SENTIMENT_COLORS['negative'],
            SENTIMENT_LABELS['neutral']: SENTIMENT_COLORS['neutral'],
            SENTIMENT_LABELS['positive']: SENTIMENT_COLORS['positive']
        },
        title="Sentiment by Airline",
        text='count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title="Airline", yaxis_title="Tweet Count")
    return fig

def create_timeline_chart(df):
    """Line chart for sentiment over time"""
    df['date'] = pd.to_datetime(df['date'])
    df['label'] = df['sentiment'].map(SENTIMENT_LABELS)
    
    fig = px.line(
        df,
        x='date',
        y='count',
        color='label',
        markers=True,
        color_discrete_map={
            SENTIMENT_LABELS['negative']: SENTIMENT_COLORS['negative'],
            SENTIMENT_LABELS['neutral']: SENTIMENT_COLORS['neutral'],
            SENTIMENT_LABELS['positive']: SENTIMENT_COLORS['positive']
        },
        title="Sentiment Timeline"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Tweet Count")
    return fig

def create_negative_reasons_chart(df):
    """Horizontal bar chart for negative reasons"""
    fig = px.bar(
        df,
        y='negativereason',
        x='count',
        orientation='h',
        color='count',
        color_continuous_scale='Reds',
        title="Top Negative Reasons"
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        yaxis_title="",
        xaxis_title="Count"
    )
    return fig

def create_hourly_chart(df):
    """Bar chart for tweets by hour"""
    fig = px.bar(
        df,
        x='hour',
        y='count',
        color='count',
        color_continuous_scale='Blues',
        title="Tweet Activity by Hour"
    )
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Tweet Count")
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Page config
    st.set_page_config(
        page_title="SkyPulse Analytics Dashboard",
        page_icon="✈️",
        layout="wide"
    )
    
    # Header
    st.title("✈️ SkyPulse Analytics - Airline Sentiment Dashboard")
    st.markdown("Real-time ML predictions from streaming pipeline")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Last Update**\n{datetime.now().strftime('%H:%M:%S')}")
    
    # Load metrics
    metrics_query = """
    SELECT 
        COUNT(*) as total_tweets,
        SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
        SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
        SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
        COUNT(DISTINCT airline) as airline_count
    FROM tweets_stream
    """
    
    metrics_df = load_data_from_db(metrics_query)
    
    # Check if data exists
    if metrics_df.empty or metrics_df.iloc[0]['total_tweets'] == 0:
        st.warning("⚠️ No data available yet!")
        st.info("Run the consumer script to populate data: `python streaming/consumer.py`")
        st.stop()
    
    metrics = metrics_df.iloc[0]
    total = int(metrics['total_tweets'])
    
    # ========================================================================
    # KEY METRICS ROW
    # ========================================================================
    st.header("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Tweets", f"{total:,}")
    
    with col2:
        airlines = int(metrics['airline_count'])
        st.metric("Airlines", f"{airlines}")
    
    with col3:
        positive = int(metrics['positive_count'])
        pos_pct = (positive / total * 100) if total > 0 else 0
        st.metric("Positive 😊", f"{pos_pct:.1f}%", f"{positive} tweets")
    
    with col4:
        neutral = int(metrics['neutral_count'])
        neu_pct = (neutral / total * 100) if total > 0 else 0
        st.metric("Neutral 😐", f"{neu_pct:.1f}%", f"{neutral} tweets")
    
    with col5:
        negative = int(metrics['negative_count'])
        neg_pct = (negative / total * 100) if total > 0 else 0
        st.metric("Negative 😞", f"{neg_pct:.1f}%", f"{negative} tweets")
    
    st.markdown("---")
    
    # ========================================================================
    # CHARTS ROW 1: Sentiment Pie + Airline Bar
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_query = """
        SELECT sentiment, COUNT(*) as count
        FROM tweets_stream
        GROUP BY sentiment
        ORDER BY sentiment
        """
        sentiment_df = load_data_from_db(sentiment_query)
        
        if not sentiment_df.empty:
            fig = create_sentiment_pie_chart(sentiment_df)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        airline_query = """
        SELECT airline, sentiment, COUNT(*) as count
        FROM tweets_stream
        GROUP BY airline, sentiment
        ORDER BY airline, sentiment
        """
        airline_df = load_data_from_db(airline_query)
        
        if not airline_df.empty:
            fig = create_airline_bar_chart(airline_df)
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # CHARTS ROW 2: Timeline + Hourly Activity
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        timeline_query = """
        SELECT 
            DATE(tweet_created) as date,
            sentiment,
            COUNT(*) as count
        FROM tweets_stream
        WHERE tweet_created IS NOT NULL
        GROUP BY DATE(tweet_created), sentiment
        ORDER BY date, sentiment
        """
        timeline_df = load_data_from_db(timeline_query)
        
        if not timeline_df.empty:
            fig = create_timeline_chart(timeline_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available")
    
    with col2:
        hourly_query = """
        SELECT 
            EXTRACT(HOUR FROM inserted_at) as hour,
            COUNT(*) as count
        FROM tweets_stream
        GROUP BY EXTRACT(HOUR FROM inserted_at)
        ORDER BY hour
        """
        hourly_df = load_data_from_db(hourly_query)
        
        if not hourly_df.empty:
            fig = create_hourly_chart(hourly_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data available")
    
    # ========================================================================
    # CHARTS ROW 3: Negative Reasons
    # ========================================================================
    st.subheader("📉 Negative Feedback Analysis")
    
    negative_query = """
    SELECT negativereason, COUNT(*) as count
    FROM tweets_stream
    WHERE sentiment = 'negative' 
        AND negativereason IS NOT NULL 
        AND negativereason != ''
    GROUP BY negativereason
    ORDER BY count DESC
    LIMIT 10
    """
    negative_df = load_data_from_db(negative_query)
    
    if not negative_df.empty:
        fig = create_negative_reasons_chart(negative_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No negative reasons recorded yet")
    
    # ========================================================================
    # DATA TABLE
    # ========================================================================
    st.markdown("---")
    st.subheader("📋 Recent Tweets")
    
    tweets_query = """
    SELECT 
        airline,
        sentiment,
        text,
        negativereason,
        tweet_created,
        inserted_at
    FROM tweets_stream
    ORDER BY inserted_at DESC
    LIMIT 100
    """
    tweets_df = load_data_from_db(tweets_query)
    
    if not tweets_df.empty:
        # Add emoji labels
        tweets_df['sentiment_label'] = tweets_df['sentiment'].map(SENTIMENT_LABELS)
        
        # Display table
        display_df = tweets_df[['airline', 'sentiment_label', 'text', 'negativereason', 'tweet_created']]
        display_df.columns = ['Airline', 'Sentiment', 'Text', 'Negative Reason', 'Created At']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = tweets_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Data (CSV)",
            csv,
            f"tweets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("*SkyPulse Analytics - Real-time Airline Sentiment Analysis*")

if __name__ == "__main__":
    main()
