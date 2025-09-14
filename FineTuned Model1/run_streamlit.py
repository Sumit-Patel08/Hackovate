"""
Script to run the Streamlit dashboard for cattle milk yield prediction.
"""

import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    print("🚀 Starting Streamlit dashboard for AI/ML Cattle Milk Yield Prediction...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    # Set the path to the streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")
    
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())
