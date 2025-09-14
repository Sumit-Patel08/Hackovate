"""
Streamlit Dashboard for AI/ML-Based Cattle Milk Yield Prediction System
Interactive farmer interface for Model 1: Comprehensive milk yield prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import MilkYieldPredictor, create_sample_cow_data

# Page configuration
st.set_page_config(
    page_title="🐄 Cattle Milk Yield Prediction Dashboard",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f8f0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2E8B57;
}
.prediction-result {
    background-color: #e8f5e8;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 2px solid #2E8B57;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load the prediction model (cached)."""
    try:
        predictor = MilkYieldPredictor()
        return predictor if predictor.model is not None else None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🐄 AI/ML Cattle Milk Yield Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.predictor is None:
        with st.spinner("Loading prediction model..."):
            st.session_state.predictor = load_model()
            st.session_state.model_loaded = st.session_state.predictor is not None
    
    if not st.session_state.model_loaded:
        st.error("⚠️ Model not available. Please train the model first by running `python train_model.py`")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Single Prediction", "Batch Prediction", "Farm Analytics", "Model Information", "Reports"]
    )
    
    # Display pages
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Farm Analytics":
        farm_analytics_page()
    elif page == "Model Information":
        model_info_page()
    elif page == "Reports":
        reports_page()

def single_prediction_page():
    """Single cow milk yield prediction page."""
    st.header("🎯 Single Cow Milk Yield Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Cow Information")
        
        with st.form("prediction_form"):
            # Animal data section
            st.markdown("### 🐄 Animal Information")
            col_a, col_b = st.columns(2)
            
            with col_a:
                cow_id = st.text_input("Cow ID", value="COW001")
                age_months = st.number_input("Age (months)", min_value=12, max_value=200, value=48, step=1)
                weight_kg = st.number_input("Weight (kg)", min_value=300.0, max_value=1200.0, value=580.0, step=5.0)
                breed = st.selectbox("Breed", ["Holstein", "Jersey", "Guernsey", "Ayrshire", "Brown Swiss", "Simmental"])
            
            with col_b:
                lactation_stage = st.selectbox("Lactation Stage", ["early", "peak", "mid", "late", "dry"], index=1)
                lactation_day = st.number_input("Lactation Day", min_value=1, max_value=365, value=60)
                parity = st.number_input("Parity (calvings)", min_value=1, max_value=10, value=3)
                historical_yield_7d = st.number_input("Avg Yield Last 7 Days (L)", min_value=0.0, value=32.5, step=0.1)
            
            # Feed and nutrition section
            st.markdown("### 🌾 Feed & Nutrition")
            col_c, col_d = st.columns(2)
            
            with col_c:
                feed_type = st.selectbox("Feed Type", ["green_fodder", "dry_fodder", "concentrates", "silage", "mixed"])
                feed_quantity_kg = st.number_input("Feed Quantity (kg/day)", min_value=5.0, max_value=25.0, value=14.5, step=0.1)
            
            with col_d:
                feeding_frequency = st.number_input("Feeding Frequency (times/day)", min_value=1, max_value=6, value=3)
                historical_yield_30d = st.number_input("Avg Yield Last 30 Days (L)", min_value=0.0, value=31.8, step=0.1)
            
            # Activity and behavior section
            st.markdown("### 🚶 Activity & Behavior")
            col_e, col_f = st.columns(2)
            
            with col_e:
                walking_distance_km = st.number_input("Walking Distance (km/day)", min_value=0.0, max_value=15.0, value=4.2, step=0.1)
                grazing_hours = st.number_input("Grazing Hours", min_value=0.0, max_value=12.0, value=7.5, step=0.1)
            
            with col_f:
                rumination_hours = st.number_input("Rumination Hours", min_value=4.0, max_value=12.0, value=8.2, step=0.1)
                resting_hours = st.number_input("Resting Hours", min_value=6.0, max_value=16.0, value=8.3, step=0.1)
            
            # Health data section
            st.markdown("### 🏥 Health Data")
            col_g, col_h = st.columns(2)
            
            with col_g:
                body_temperature = st.number_input("Body Temperature (°C)", min_value=36.0, max_value=42.0, value=38.6, step=0.1)
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=100.0, value=65.0, step=1.0)
            
            with col_h:
                health_score = st.slider("Health Score", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
            
            # Environmental data section
            st.markdown("### 🌡️ Environmental Conditions")
            col_i, col_j = st.columns(2)
            
            with col_i:
                temperature = st.number_input("Ambient Temperature (°C)", min_value=-20.0, max_value=50.0, value=22.0, step=0.1)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
                season = st.selectbox("Season", ["spring", "summer", "autumn", "winter"], index=1)
            
            with col_j:
                housing_type = st.selectbox("Housing Type", ["free_stall", "tie_stall", "pasture", "compost_barn"])
                ventilation_score = st.slider("Ventilation Quality", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
                cleanliness_score = st.slider("Cleanliness Score", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
            
            day_of_year = st.number_input("Day of Year", min_value=1, max_value=365, value=180)
            
            submitted = st.form_submit_button("🔮 Predict Milk Yield", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = {
                    'age_months': age_months,
                    'weight_kg': weight_kg,
                    'breed': breed,
                    'lactation_stage': lactation_stage,
                    'lactation_day': lactation_day,
                    'parity': parity,
                    'historical_yield_7d': historical_yield_7d,
                    'historical_yield_30d': historical_yield_30d,
                    'feed_type': feed_type,
                    'feed_quantity_kg': feed_quantity_kg,
                    'feeding_frequency': feeding_frequency,
                    'walking_distance_km': walking_distance_km,
                    'grazing_hours': grazing_hours,
                    'rumination_hours': rumination_hours,
                    'resting_hours': resting_hours,
                    'body_temperature': body_temperature,
                    'heart_rate': heart_rate,
                    'health_score': health_score,
                    'temperature': temperature,
                    'humidity': humidity,
                    'season': season,
                    'housing_type': housing_type,
                    'ventilation_score': ventilation_score,
                    'cleanliness_score': cleanliness_score,
                    'day_of_year': day_of_year
                }
                
                # Validate and predict
                with st.spinner("Making prediction..."):
                    validation = st.session_state.predictor.validate_input(input_data)
                    result = st.session_state.predictor.predict_single_cow(input_data)
                
                # Store results
                st.session_state.last_prediction = {
                    "cow_id": cow_id,
                    "input": input_data,
                    "result": result,
                    "validation": validation
                }
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'last_prediction' in st.session_state:
            pred_data = st.session_state.last_prediction
            
            if "error" in pred_data["result"]:
                st.error(f"❌ Prediction failed: {pred_data['result']['error']}")
            else:
                predicted_yield = pred_data["result"]["predicted_milk_yield"]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-result">
                    🐄 {pred_data['cow_id']}<br>
                    Predicted Milk Yield<br>
                    <span style="font-size: 2rem; color: #2E8B57;">{predicted_yield:.2f} L</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Validation warnings
                if pred_data["validation"]["warnings"]:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning("⚠️ Validation Warnings:")
                    for warning in pred_data["validation"]["warnings"]:
                        st.write(f"• {warning}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Performance insights
                st.markdown("### 📊 Performance Insights")
                
                if predicted_yield > 30:
                    st.success("🟢 Excellent milk yield expected!")
                    st.write("This cow is performing above average.")
                elif predicted_yield > 20:
                    st.info("🟡 Good milk yield expected.")
                    st.write("Performance is within normal range.")
                else:
                    st.warning("🔴 Below average yield predicted.")
                    st.write("Consider reviewing feed and health factors.")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                recommendations = generate_recommendations(pred_data["input"], predicted_yield)
                for rec in recommendations:
                    st.write(f"• {rec}")
        else:
            st.info("Enter cow information and click 'Predict' to see results.")

def batch_prediction_page():
    """Batch prediction page for multiple cows."""
    st.header("📊 Batch Milk Yield Prediction")
    
    # File upload section
    st.subheader("📁 Upload Cow Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head(10))
            
            # Validate required columns
            required_cols = ['age_months', 'weight_kg', 'breed', 'lactation_stage', 
                           'feed_type', 'feed_quantity_kg', 'temperature', 'humidity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV contains all required columns.")
            else:
                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        result = st.session_state.predictor.predict_batch(df)
                        
                        if "error" in result:
                            st.error(f"❌ Batch prediction failed: {result['error']}")
                        else:
                            # Add predictions to dataframe
                            df['predicted_milk_yield'] = result["predictions"]
                            
                            # Display results
                            st.success(f"✅ Processed {result['count']} predictions successfully!")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Cows", len(df))
                            with col2:
                                st.metric("Avg Yield", f"{np.mean(result['predictions']):.2f} L")
                            with col3:
                                st.metric("Max Yield", f"{np.max(result['predictions']):.2f} L")
                            with col4:
                                st.metric("Min Yield", f"{np.min(result['predictions']):.2f} L")
                            
                            # Results table
                            st.subheader("📋 Prediction Results")
                            st.dataframe(df)
                            
                            # Visualization
                            create_batch_visualizations(df)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"milk_yield_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    
    else:
        # Sample template
        st.subheader("📝 Sample Data Template")
        st.info("Download this template to see the required format for batch predictions.")
        
        sample_data = pd.DataFrame([
            create_sample_cow_data(),
            {**create_sample_cow_data(), "age_months": 36, "breed": "Jersey"},
            {**create_sample_cow_data(), "lactation_stage": "late", "feed_quantity_kg": 10.5}
        ])
        
        st.dataframe(sample_data)
        
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Template",
            data=csv_sample,
            file_name="cattle_data_template.csv",
            mime="text/csv"
        )

def farm_analytics_page():
    """Farm analytics and insights page."""
    st.header("📈 Farm Analytics & Insights")
    
    # Generate sample farm data for demonstration
    if st.button("🔄 Generate Sample Farm Data"):
        with st.spinner("Generating farm analytics..."):
            farm_data = generate_sample_farm_data()
            st.session_state.farm_data = farm_data
    
    if 'farm_data' in st.session_state:
        df = st.session_state.farm_data
        
        # Farm overview metrics
        st.subheader("🏪 Farm Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cows", len(df))
        with col2:
            st.metric("Avg Daily Yield", f"{df['predicted_milk_yield'].mean():.1f} L")
        with col3:
            st.metric("Total Daily Production", f"{df['predicted_milk_yield'].sum():.0f} L")
        with col4:
            st.metric("Top Performer", f"{df['predicted_milk_yield'].max():.1f} L")
        
        # Visualizations
        create_farm_analytics_charts(df)
        
        # Performance insights
        st.subheader("💡 Farm Insights")
        
        # Top and bottom performers
        top_performers = df.nlargest(5, 'predicted_milk_yield')[['cow_id', 'breed', 'predicted_milk_yield']]
        bottom_performers = df.nsmallest(5, 'predicted_milk_yield')[['cow_id', 'breed', 'predicted_milk_yield']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🏆 Top 5 Performers**")
            st.dataframe(top_performers)
        
        with col2:
            st.write("**⚠️ Attention Needed**")
            st.dataframe(bottom_performers)

def model_info_page():
    """Model information and performance page."""
    st.header("🤖 Model Information")
    
    # Model performance
    model_info = st.session_state.predictor.get_model_info()
    
    if "error" not in model_info:
        st.subheader("📊 Model Performance")
        
        # Display performance metrics
        performance = model_info["model_performance"]
        
        # Create performance comparison chart
        models = list(performance.keys())
        r2_scores = [performance[m]['test_r2'] for m in models]
        rmse_scores = [performance[m]['test_rmse'] for m in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score by Model', 'RMSE by Model')
        )
        
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name="R² Score"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=rmse_scores, name="RMSE"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model info
        best_model = max(performance.items(), key=lambda x: x[1]['test_r2'])
        st.success(f"🏆 Best Model: **{best_model[0]}** (R² = {best_model[1]['test_r2']:.4f})")
    
    # Feature information
    st.subheader("📋 Model Features")
    
    feature_descriptions = st.session_state.predictor._get_feature_descriptions()
    
    # Group features by category
    categories = {
        "Animal Data": ["age_months", "weight_kg", "breed", "lactation_stage", "lactation_day", "parity"],
        "Feed & Nutrition": ["feed_type", "feed_quantity_kg", "feeding_frequency"],
        "Activity & Behavior": ["walking_distance_km", "grazing_hours", "rumination_hours", "resting_hours"],
        "Health Data": ["body_temperature", "heart_rate", "health_score"],
        "Environmental": ["temperature", "humidity", "season", "housing_type", "ventilation_score", "cleanliness_score"]
    }
    
    for category, features in categories.items():
        with st.expander(f"**{category}** ({len(features)} features)"):
            for feature in features:
                if feature in feature_descriptions:
                    st.write(f"• **{feature}**: {feature_descriptions[feature]}")

def reports_page():
    """Reports and export page."""
    st.header("📋 Reports & Export")
    
    st.subheader("📊 Generate Farm Report")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", ["Daily Summary", "Weekly Analysis", "Monthly Overview"])
        include_charts = st.checkbox("Include Charts", value=True)
    
    with col2:
        date_range = st.date_input("Report Date", value=datetime.now().date())
        export_format = st.selectbox("Export Format", ["PDF", "Excel", "CSV"])
    
    if st.button("📄 Generate Report"):
        with st.spinner("Generating report..."):
            # Generate sample report data
            if 'farm_data' not in st.session_state:
                st.session_state.farm_data = generate_sample_farm_data()
            
            df = st.session_state.farm_data
            
            # Create report summary
            st.subheader(f"📈 {report_type} - {date_range}")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Production", f"{df['predicted_milk_yield'].sum():.0f} L")
            with col2:
                st.metric("Average per Cow", f"{df['predicted_milk_yield'].mean():.1f} L")
            with col3:
                st.metric("Production Efficiency", f"{(df['predicted_milk_yield'].mean() / 30 * 100):.1f}%")
            
            # Export data
            if export_format == "CSV":
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name=f"farm_report_{date_range}.csv",
                    mime="text/csv"
                )
            
            st.success("✅ Report generated successfully!")

def generate_recommendations(input_data, predicted_yield):
    """Generate recommendations based on input data and prediction."""
    recommendations = []
    
    if input_data["feed_quantity_kg"] < 10:
        recommendations.append("Consider increasing feed quantity for better milk production")
    
    if input_data["health_score"] < 0.8:
        recommendations.append("Monitor cow health closely - low health score detected")
    
    if input_data["body_temperature"] > 39.5:
        recommendations.append("Elevated body temperature - check for fever or heat stress")
    
    if input_data["walking_distance_km"] < 2:
        recommendations.append("Encourage more physical activity for better health")
    
    if input_data["temperature"] > 30:
        recommendations.append("Provide cooling measures - high ambient temperature")
    
    if predicted_yield < 20:
        recommendations.append("Consider veterinary consultation for low yield prediction")
    
    if not recommendations:
        recommendations.append("Current conditions look optimal for milk production")
    
    return recommendations

def create_batch_visualizations(df):
    """Create visualizations for batch prediction results."""
    st.subheader("📊 Batch Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Yield distribution
        fig1 = px.histogram(df, x='predicted_milk_yield', 
                           title="Distribution of Predicted Milk Yields",
                           labels={'predicted_milk_yield': 'Predicted Milk Yield (L)'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Yield by breed
        if 'breed' in df.columns:
            breed_avg = df.groupby('breed')['predicted_milk_yield'].mean().reset_index()
            fig2 = px.bar(breed_avg, x='breed', y='predicted_milk_yield',
                         title="Average Yield by Breed")
            st.plotly_chart(fig2, use_container_width=True)

def generate_sample_farm_data():
    """Generate sample farm data for analytics."""
    np.random.seed(42)
    n_cows = 50
    
    data = []
    for i in range(n_cows):
        cow_data = create_sample_cow_data()
        cow_data['cow_id'] = f"COW{i+1:03d}"
        
        # Add some variation
        cow_data['age_months'] = np.random.randint(24, 120)
        cow_data['breed'] = np.random.choice(['Holstein', 'Jersey', 'Guernsey'])
        cow_data['feed_quantity_kg'] = np.random.uniform(8, 16)
        
        # Predict yield
        result = st.session_state.predictor.predict_single_cow(cow_data)
        cow_data['predicted_milk_yield'] = result.get('predicted_milk_yield', 0)
        
        data.append(cow_data)
    
    return pd.DataFrame(data)

def create_farm_analytics_charts(df):
    """Create comprehensive farm analytics charts."""
    
    # Yield distribution by breed
    fig1 = px.box(df, x='breed', y='predicted_milk_yield',
                  title="Milk Yield Distribution by Breed")
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs Yield
        fig2 = px.scatter(df, x='age_months', y='predicted_milk_yield',
                         color='breed', title="Age vs Milk Yield")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Feed quantity vs Yield
        fig3 = px.scatter(df, x='feed_quantity_kg', y='predicted_milk_yield',
                         color='lactation_stage', title="Feed Quantity vs Milk Yield")
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
