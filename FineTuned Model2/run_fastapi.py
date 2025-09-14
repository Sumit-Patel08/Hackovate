"""
Script to run FastAPI server for Cattle Disease Detection Model
"""

import uvicorn
import os
import sys

def main():
    """Run the FastAPI server."""
    print("Starting Cattle Disease Detection API Server...")
    print("Server will be available at: http://localhost:8001")
    print("API Documentation: http://localhost:8001/docs")
    print("="*50)
    
    # Check if model exists
    model_path = "models/cattle_disease_classifier.joblib"
    if not os.path.exists(model_path):
        print("WARNING: Trained model not found!")
        print("Please run training first: python run_training.py")
        print("The API will start but predictions will not work until model is trained.")
        print("="*50)
    
    try:
        uvicorn.run(
            "app.fastapi_app:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
