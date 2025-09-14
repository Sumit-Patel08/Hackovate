"""
Script to run the FastAPI server for cattle milk yield prediction.
"""

import uvicorn
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting FastAPI server for AI/ML Cattle Milk Yield Prediction...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "app.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
