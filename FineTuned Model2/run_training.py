"""
Script to run the complete training pipeline for Cattle Disease Detection Model
"""

import os
import sys
from train_model import main

if __name__ == "__main__":
    print("Starting Cattle Disease Detection Model Training Pipeline...")
    print("This will generate data, train models, and save the best performing model.")
    print("="*60)
    
    try:
        main()
        print("\n" + "="*60)
        print("SUCCESS: Training completed successfully!")
        print("You can now run the FastAPI server using: python -m uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8001 --reload")
    except Exception as e:
        print(f"\nERROR: Training failed with error: {e}")
        sys.exit(1)
