import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')

# Import our custom modules
from models.ml_predictor import MarketPredictor
from services.data_service import DataService
from services.openai_service import OpenAIAnalysisService
from services.government_schemes import GovernmentSchemeService
from utils.data_processor import DataProcessor
from config.settings import CROP_CATEGORIES, INDIAN_STATES

# Page configuration
st.set_page_config(
    page_title="AgriMarket AI - Agricultural Market Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def initialize_services():
    return {
        'data_service': DataService(),
        'openai_service': OpenAIAnalysisService(),
        'government_service': GovernmentSchemeService(),
        'data_processor': DataProcessor(),
        'predictor': MarketPredictor()
    }

services = initialize_services()

def text_to_speech_component(text_to_read, button_label="🔊 Read Aloud", section_id="default"):
    """Add a text-to-speech button that reads the provided text"""
    button_id = f"tts_button_{section_id}_{hash(text_to_read + section_id) % 10000}"
    
    # Clean text for speech (remove markdown and special characters)
    clean_text = text_to_read.replace("*", "").replace("#", "").replace("**", "").replace("₹", "rupees ")
    clean_text = clean_text.replace("🌾", "").replace("📊", "").replace("🧠", "").replace("🏛️", "")
    clean_text = clean_text.replace("🎯", "").replace("⚠️", "").replace("🚀", "").replace("📈", "")
    
    if st.button(button_label, key=button_id):
        # JavaScript for text-to-speech
        tts_html = f"""
        <script>
        function speakText() {{
            const text = `{clean_text}`;
            if ('speechSynthesis' in window) {{
                // Cancel any ongoing speech
                window.speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 0.8;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                
                // Set voice (prefer English voices)
                const voices = window.speechSynthesis.getVoices();
                const englishVoice = voices.find(voice => voice.lang.startsWith('en'));
                if (englishVoice) {{
                    utterance.voice = englishVoice;
                }}
                
                window.speechSynthesis.speak(utterance);
            }} else {{
                alert('Text-to-speech is not supported in your browser');
            }}
        }}
        
        // Wait for voices to load, then speak
        if (window.speechSynthesis.getVoices().length > 0) {{
            speakText();
        }} else {{
            window.speechSynthesis.addEventListener('voiceschanged', function() {{
                speakText();
            }});
        }}
        </script>
        """
        components.html(tts_html, height=0)

def stop_speech_component():
    """Add a stop speech button"""
    if st.button("🔇 Stop Reading", key="stop_speech"):
        stop_html = """
        <script>
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        </script>
        """
        components.html(stop_html, height=0)

def main():
    # App header
    st.title("🌾 AgriMarket AI")
    st.markdown("### ML-Powered Agricultural Market Analysis & Trend Prediction")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Market Analysis Parameters")
        
        # Crop selection
        crop_category = st.selectbox(
            "Select Crop Category",
            options=list(CROP_CATEGORIES.keys()),
            help="Choose the category of your crop"
        )
        
        crop = st.selectbox(
            "Select Specific Crop",
            options=CROP_CATEGORIES[crop_category],
            help="Select the specific crop for analysis"
        )
        
        # Location selection
        location = st.selectbox(
            "Select State/Location",
            options=INDIAN_STATES,
            help="Choose your state for location-specific analysis"
        )
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        show_predictions = st.checkbox("Price Predictions", value=True)
        show_ai_analysis = st.checkbox("AI Market Insights", value=True)
        show_government_schemes = st.checkbox("Government Schemes", value=True)
        show_technical_analysis = st.checkbox("Technical Analysis", value=True)
        
        # Analyze button
        analyze_button = st.button("🚀 Analyze Market", type="primary", use_container_width=True)
    
    # Main content area
    if analyze_button or 'analysis_complete' in st.session_state:
        st.session_state.analysis_complete = True
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch market data
            status_text.text("📊 Fetching market data...")
            progress_bar.progress(20)
            
            market_data = services['data_service'].get_market_data(crop, location)
            weather_data = services['data_service'].get_weather_data(location)
            
            # Step 2: Generate ML predictions
            status_text.text("🤖 Training ML models and generating predictions...")
            progress_bar.progress(40)
            
            historical_data = services['data_processor'].generate_historical_data(crop, location, days=365)
            X, y = services['data_processor'].prepare_features(historical_data)
            
            # Train models
            model_scores = services['predictor'].train_models(X, y)
            
            # Generate predictions
            latest_features = X[-1:] if len(X) > 0 else np.zeros((1, 9))
            future_predictions = services['predictor'].predict_future_prices(latest_features[0], days=3)
            
            # Step 3: AI Analysis
            status_text.text("🧠 Generating AI insights...")
            progress_bar.progress(60)
            
            ai_analysis = {}
            ai_recommendations = {}
            price_factors = {}
            
            if show_ai_analysis:
                ai_analysis = services['openai_service'].analyze_market_trends(
                    crop, location, market_data, future_predictions
                )
                ai_recommendations = services['openai_service'].get_farming_recommendations(
                    crop, location, ai_analysis
                )
                price_factors = services['openai_service'].analyze_price_factors(
                    crop, location, market_data
                )
            
            # Step 4: Government schemes
            status_text.text("🏛️ Finding relevant government schemes...")
            progress_bar.progress(80)
            
            relevant_schemes = []
            if show_government_schemes:
                relevant_schemes = services['government_service'].get_relevant_schemes(crop, location)
            
            # Step 5: Complete analysis
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            display_analysis_results(
                crop, location, market_data, weather_data, historical_data,
                future_predictions, model_scores, ai_analysis, ai_recommendations,
                price_factors, relevant_schemes, show_predictions, show_ai_analysis,
                show_government_schemes, show_technical_analysis
            )
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")
    
    else:
        # Welcome screen
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen with app information"""
    st.markdown("---")
    
    # Add read aloud for welcome screen
    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown("### Welcome to AgriMarket AI")
    with col2:
        welcome_text = "Welcome to AgriMarket AI. Select your crop and location to get current price, tomorrow prediction, volume traded, and market trend."
        text_to_speech_component(welcome_text, "🔊 Read Introduction", "welcome")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 **Market Predictions**
        - ML-powered price forecasting
        - Yesterday vs Today vs Tomorrow analysis
        - Confidence scoring and risk assessment
        - Technical indicators and trends
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 **AI Insights**
        - OpenAI-powered market analysis
        - Intelligent trend interpretation
        - Risk factors and opportunities
        - Personalized recommendations
        """)
    
    with col3:
        st.markdown("""
        ### 🏛️ **Government Support**
        - Relevant scheme recommendations
        - Application guidance
        - Eligibility criteria
        - Contact information
        """)
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("### 🌟 Key Features")
    
    features = [
        "📈 **Real-time Market Data**: Access current prices and trading volumes",
        "🤖 **ML Price Prediction**: Advanced algorithms for accurate forecasting",
        "🧠 **AI Market Analysis**: OpenAI-powered insights and recommendations",
        "📊 **Interactive Charts**: Dynamic visualizations of price trends",
        "🏛️ **Government Schemes**: Personalized scheme recommendations",
        "📱 **User-friendly Interface**: Easy-to-use Streamlit web application"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    st.markdown("---")
    st.info("👈 Select your crop and location from the sidebar to start the analysis!")

def display_analysis_results(crop, location, market_data, weather_data, historical_data,
                           future_predictions, model_scores, ai_analysis, ai_recommendations,
                           price_factors, relevant_schemes, show_predictions, show_ai_analysis,
                           show_government_schemes, show_technical_analysis):
    """Display comprehensive analysis results"""
    
    # Market summary header
    st.markdown("---")
    st.header(f"📊 Market Analysis: {crop} in {location}")
    
    # Generate market summary
    if show_ai_analysis and ai_analysis:
        summary_data = {
            'market_data': market_data,
            'predictions': future_predictions,
            'analysis': ai_analysis
        }
        market_summary = services['openai_service'].generate_market_summary(crop, location, summary_data)
        
        col1, col2, col3 = st.columns([6, 1, 1])
        with col1:
            st.info(f"**Market Summary**: {market_summary}")
        with col2:
            # Read only key metrics: current price, tomorrow prediction, volume traded, market trend
            current_price = market_data.get('current_price', 0)
            tomorrow_price = future_predictions[0] if future_predictions else current_price
            volume = market_data.get('volume_traded', 0)
            trend = market_data.get('trend', 'Stable')
            
            key_metrics_text = f"Current price: {current_price:.0f} rupees. Tomorrow prediction: {tomorrow_price:.0f} rupees. Volume traded: {volume:.0f} tonnes. Market trend: {trend}."
            text_to_speech_component(key_metrics_text, "🔊 Read Key Metrics", "summary")
        with col3:
            stop_speech_component()
    
    # Key metrics cards
    display_key_metrics(market_data, weather_data, future_predictions)
    
    # Price comparison table
    if show_predictions:
        display_price_comparison_table(market_data, future_predictions)
    
    # Charts section
    st.markdown("---")
    st.header("📈 Price Analysis & Trends")
    
    # Price trend chart
    display_price_trend_chart(historical_data, future_predictions)
    
    if show_technical_analysis:
        # Technical analysis charts
        display_technical_analysis_charts(historical_data)
        
        # Model performance
        display_model_performance(model_scores)
    
    # AI Analysis section
    if show_ai_analysis and ai_analysis:
        st.markdown("---")
        col1, col2 = st.columns([6, 2])
        with col1:
            st.header("🧠 AI Market Insights")
        with col2:
            # Read only key metrics: current price, tomorrow prediction, volume traded, market trend
            current_price = market_data.get('current_price', 0)
            tomorrow_price = future_predictions[0] if future_predictions else current_price
            volume = market_data.get('volume_traded', 0)
            trend = market_data.get('trend', 'Stable')
            
            metrics_text = f"Current price: {current_price:.0f} rupees. Tomorrow prediction: {tomorrow_price:.0f} rupees. Volume traded: {volume:.0f} tonnes. Market trend: {trend}."
            text_to_speech_component(metrics_text, "🔊 Read Metrics", "ai_insights")
        
        display_ai_analysis(ai_analysis, ai_recommendations, price_factors)
    
    # Government schemes section
    if show_government_schemes and relevant_schemes:
        st.markdown("---")
        col1, col2 = st.columns([6, 2])
        with col1:
            st.header("🏛️ Government Schemes & Support")
        with col2:
            # Read only key metrics: current price, tomorrow prediction, volume traded, market trend
            current_price = market_data.get('current_price', 0)
            tomorrow_price = future_predictions[0] if future_predictions else current_price
            volume = market_data.get('volume_traded', 0)
            trend = market_data.get('trend', 'Stable')
            
            metrics_text = f"Current price: {current_price:.0f} rupees. Tomorrow prediction: {tomorrow_price:.0f} rupees. Volume traded: {volume:.0f} tonnes. Market trend: {trend}."
            text_to_speech_component(metrics_text, "🔊 Read Metrics", "schemes")
        
        display_government_schemes(relevant_schemes)
    
    # Market news and updates
    display_market_news(crop)

def display_key_metrics(market_data, weather_data, future_predictions):
    """Display key market metrics in card format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = market_data.get('current_price', 0)
        yesterday_price = market_data.get('yesterday_price', current_price)
        price_change = current_price - yesterday_price
        price_change_pct = (price_change / yesterday_price * 100) if yesterday_price > 0 else 0
        
        st.metric(
            label="Current Price",
            value=f"₹{current_price:,.0f}",
            delta=f"{price_change_pct:+.1f}%"
        )
    
    with col2:
        tomorrow_price = future_predictions[0] if future_predictions else current_price
        tomorrow_change = ((tomorrow_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        st.metric(
            label="Tomorrow Prediction",
            value=f"₹{tomorrow_price:,.0f}",
            delta=f"{tomorrow_change:+.1f}%"
        )
    
    with col3:
        volume = market_data.get('volume_traded', 0)
        st.metric(
            label="Volume Traded",
            value=f"{volume:,.0f} tonnes"
        )
    
    with col4:
        trend = market_data.get('trend', 'Stable')
        trend_color = {"Bullish": "🟢", "Bearish": "🔴", "Stable": "🟡"}.get(trend, "⚪")
        st.metric(
            label="Market Trend",
            value=f"{trend_color} {trend}"
        )

def display_price_comparison_table(market_data, future_predictions):
    """Display yesterday vs today vs tomorrow price comparison"""
    col1, col2 = st.columns([6, 2])
    with col1:
        st.subheader("📅 Price Comparison Table")
    with col2:
        # Read only key metrics: current price, tomorrow prediction, volume traded, market trend
        current_price = market_data.get('current_price', 0)
        tomorrow_price = future_predictions[0] if future_predictions else current_price
        volume = market_data.get('volume_traded', 0)
        trend = market_data.get('trend', 'Stable')
        
        metrics_text = f"Current price: {current_price:.0f} rupees. Tomorrow prediction: {tomorrow_price:.0f} rupees. Volume traded: {volume:.0f} tonnes. Market trend: {trend}."
        text_to_speech_component(metrics_text, "🔊 Read Metrics", "price_table")
    
    current_price = market_data.get('current_price', 0)
    yesterday_price = market_data.get('yesterday_price', current_price)
    
    comparison_data = {
        'Period': ['Yesterday', 'Today', 'Tomorrow'],
        'Price (₹)': [
            f"₹{yesterday_price:,.0f}",
            f"₹{current_price:,.0f}",
            f"₹{future_predictions[0]:,.0f}" if future_predictions else f"₹{current_price:,.0f}"
        ],
        'Change': [
            "-",
            f"{((current_price - yesterday_price) / yesterday_price * 100):+.1f}%" if yesterday_price > 0 else "0.0%",
            f"{((future_predictions[0] - current_price) / current_price * 100):+.1f}%" if future_predictions and current_price > 0 else "0.0%"
        ],
        'Recommendation': [
            "Historical data",
            "Current market rate",
            "Hold" if not future_predictions else ("Buy" if future_predictions[0] > current_price else "Sell")
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

def display_price_trend_chart(historical_data, future_predictions):
    """Display interactive price trend chart"""
    st.subheader("📈 Price Trend Analysis")
    
    # Create the main price chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Trend', 'Volume & Volatility'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['price'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['price_ma_7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='orange', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['price_ma_30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='red', width=1, dash='dot')
        ),
        row=1, col=1
    )
    
    # Future predictions
    if future_predictions:
        future_dates = [historical_data['date'].iloc[-1] + timedelta(days=i+1) for i in range(len(future_predictions))]
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='green', width=3, dash='dashdot'),
                marker=dict(size=8, color='green')
            ),
            row=1, col=1
        )
    
    # Volume chart
    if 'volume_traded' in historical_data.columns:
        fig.add_trace(
            go.Bar(
                x=historical_data['date'],
                y=historical_data.get('volume_traded', [1000] * len(historical_data)),
                name='Volume',
                marker_color='lightblue',
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # Volatility
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['volatility'],
            mode='lines',
            name='Volatility',
            line=dict(color='purple', width=2),
            yaxis='y3'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Price Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_analysis_charts(historical_data):
    """Display technical analysis indicators"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Distribution")
        fig_hist = px.histogram(
            historical_data, 
            x='price', 
            nbins=30,
            title="Price Distribution (Last 365 Days)"
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("📈 Price vs Moving Averages")
        
        # Calculate latest values for display
        latest_price = historical_data['price'].iloc[-1]
        latest_ma7 = historical_data['price_ma_7'].iloc[-1]
        latest_ma30 = historical_data['price_ma_30'].iloc[-1]
        
        ma_data = {
            'Indicator': ['Current Price', '7-Day MA', '30-Day MA'],
            'Value': [latest_price, latest_ma7, latest_ma30],
            'Signal': [
                'Current',
                'Bullish' if latest_price > latest_ma7 else 'Bearish',
                'Bullish' if latest_price > latest_ma30 else 'Bearish'
            ]
        }
        
        df_ma = pd.DataFrame(ma_data)
        st.dataframe(df_ma, use_container_width=True, hide_index=True)
        
        # Volatility metrics
        current_volatility = historical_data['volatility'].iloc[-1]
        avg_volatility = historical_data['volatility'].mean()
        
        st.metric(
            label="Current Volatility",
            value=f"₹{current_volatility:.0f}",
            delta=f"{((current_volatility - avg_volatility) / avg_volatility * 100):+.1f}% vs avg"
        )

def display_model_performance(model_scores):
    """Display ML model performance metrics"""
    st.subheader("🤖 ML Model Performance")
    
    if model_scores:
        performance_data = []
        for model_name, scores in model_scores.items():
            performance_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'R² Score': f"{scores['r2']:.3f}",
                'MAE': f"₹{scores['mae']:.0f}",
                'RMSE': f"₹{np.sqrt(scores['mse']):.0f}"
            })
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True, hide_index=True)
        
        # Feature importance
        feature_importance = services['predictor'].get_feature_importance()
        if feature_importance:
            st.subheader("📊 Feature Importance")
            
            importance_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=['Feature', 'Importance']
            )
            
            fig_importance = px.bar(
                importance_df.head(5),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 5 Most Important Features"
            )
            fig_importance.update_layout(height=300)
            st.plotly_chart(fig_importance, use_container_width=True)

def display_ai_analysis(ai_analysis, ai_recommendations, price_factors):
    """Display AI-powered market analysis"""
    
    # Market sentiment and insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Market Sentiment")
        sentiment = ai_analysis.get('overall_sentiment', 'Neutral')
        confidence = ai_analysis.get('confidence_score', 0)
        
        sentiment_color = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(sentiment, "⚪")
        st.markdown(f"### {sentiment_color} **{sentiment}**")
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence}%")
        
        # Key insights
        st.subheader("💡 Key Insights")
        insights = ai_analysis.get('key_insights', [])
        for insight in insights:
            st.markdown(f"• {insight}")
    
    with col2:
        st.subheader("⚠️ Risk Factors")
        risks = ai_analysis.get('risk_factors', [])
        for risk in risks:
            st.markdown(f"🔸 {risk}")
        
        st.subheader("🚀 Opportunities")
        opportunities = ai_analysis.get('opportunities', [])
        for opportunity in opportunities:
            st.markdown(f"🔹 {opportunity}")
    
    # Outlook and recommendations
    st.subheader("🔮 Market Outlook")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Short-term (1-2 weeks):**")
        st.info(ai_analysis.get('short_term_outlook', 'Market outlook unavailable'))
    
    with col2:
        st.markdown("**Medium-term (1-3 months):**")
        st.info(ai_analysis.get('medium_term_outlook', 'Market outlook unavailable'))
    
    # Recommended actions
    st.subheader("📋 Recommended Actions")
    actions = ai_analysis.get('recommended_actions', [])
    for i, action in enumerate(actions, 1):
        st.markdown(f"{i}. {action}")
    
    # Price factors analysis
    if price_factors:
        st.subheader("📊 Price Factors Analysis")
        
        tabs = st.tabs(["Supply", "Demand", "External", "Government"])
        
        with tabs[0]:
            supply_factors = price_factors.get('supply_factors', [])
            for factor in supply_factors:
                st.markdown(f"• {factor}")
        
        with tabs[1]:
            demand_factors = price_factors.get('demand_factors', [])
            for factor in demand_factors:
                st.markdown(f"• {factor}")
        
        with tabs[2]:
            external_factors = price_factors.get('external_factors', [])
            for factor in external_factors:
                st.markdown(f"• {factor}")
        
        with tabs[3]:
            gov_policies = price_factors.get('government_policies', [])
            for policy in gov_policies:
                st.markdown(f"• {policy}")
    
    # Farming recommendations
    if ai_recommendations:
        st.subheader("🌾 Farming Recommendations")
        
        rec_tabs = st.tabs(["Planting", "Harvest", "Storage", "Selling"])
        
        with rec_tabs[0]:
            st.markdown(ai_recommendations.get('planting_recommendations', 'No specific recommendations available'))
        
        with rec_tabs[1]:
            st.markdown(ai_recommendations.get('harvest_timing', 'Monitor crop maturity and market conditions'))
        
        with rec_tabs[2]:
            st.markdown(ai_recommendations.get('storage_advice', 'Follow standard storage practices'))
        
        with rec_tabs[3]:
            st.markdown(ai_recommendations.get('selling_strategy', 'Monitor market trends for optimal timing'))

def display_government_schemes(relevant_schemes):
    """Display relevant government schemes"""
    st.subheader("🏛️ Recommended Government Schemes")
    
    for scheme in relevant_schemes[:5]:  # Show top 5 schemes
        with st.expander(f"📄 {scheme['name']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Description:** {scheme['description']}")
                st.markdown(f"**Benefits:** {scheme['benefits']}")
                st.markdown(f"**Eligibility:** {scheme['eligibility']}")
            
            with col2:
                st.markdown(f"**Application Process:** {scheme['application_process']}")
                st.markdown(f"**Contact:** {scheme['contact_info']}")
                if 'website' in scheme:
                    st.markdown(f"**Website:** [{scheme['website']}]({scheme['website']})")
            
            # Application guide
            if st.button(f"📋 Get Application Guide", key=f"guide_{scheme['id']}"):
                guide = services['government_service'].get_scheme_application_guide(scheme['id'])
                if guide:
                    st.markdown("**Application Steps:**")
                    for step in guide.get('steps', []):
                        st.markdown(f"• {step}")
                    
                    st.markdown("**Required Documents:**")
                    for doc in guide.get('required_documents', []):
                        st.markdown(f"• {doc}")
                    
                    st.markdown(f"**Processing Time:** {guide.get('processing_time', 'Not specified')}")

def display_market_news(crop):
    """Display relevant market news and updates"""
    st.subheader("📰 Market News & Updates")
    
    news_items = services['data_service'].get_market_news(crop)
    
    for news in news_items:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{news['headline']}**")
                st.caption(f"{news['summary']}")
            
            with col2:
                impact_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(news['impact'], "⚪")
                st.markdown(f"{impact_color} {news['impact']}")
                st.caption(news['date'])
            
            st.markdown("---")

if __name__ == "__main__":
    main()
