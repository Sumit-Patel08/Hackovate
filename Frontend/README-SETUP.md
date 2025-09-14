# Cattle Milk Predictions - Setup Guide

## Environment Variables Setup

Create a `.env.local` file in the Frontend directory with the following variables:

```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key

# Optional: For production deployment
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## Supabase Database Setup

Run the following SQL commands in your Supabase SQL editor to set up the required tables:

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Cattle data table
CREATE TABLE cattle_data (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  cattle_id VARCHAR(50) NOT NULL,
  breed VARCHAR(100),
  age_months INTEGER,
  weight_kg DECIMAL(6,2),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Predictions table
CREATE TABLE predictions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  cattle_id VARCHAR(50),
  prediction_type VARCHAR(50), -- 'milk_yield' or 'disease_detection'
  input_data JSONB,
  prediction_result JSONB,
  confidence DECIMAL(5,4),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Reports table
CREATE TABLE reports (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  cattle_id VARCHAR(50),
  report_type VARCHAR(50) DEFAULT 'comprehensive',
  report_data JSONB,
  pdf_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE cattle_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

-- Create policies for cattle_data (only if they don't exist)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'cattle_data' AND policyname = 'Users can view own cattle data') THEN
        CREATE POLICY "Users can view own cattle data" ON cattle_data 
          FOR SELECT USING (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'cattle_data' AND policyname = 'Users can insert own cattle data') THEN
        CREATE POLICY "Users can insert own cattle data" ON cattle_data 
          FOR INSERT WITH CHECK (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'cattle_data' AND policyname = 'Users can update own cattle data') THEN
        CREATE POLICY "Users can update own cattle data" ON cattle_data 
          FOR UPDATE USING (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'cattle_data' AND policyname = 'Users can delete own cattle data') THEN
        CREATE POLICY "Users can delete own cattle data" ON cattle_data 
          FOR DELETE USING (auth.uid() = user_id);
    END IF;
END $$;

-- Create policies for predictions (only if they don't exist)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'predictions' AND policyname = 'Users can view own predictions') THEN
        CREATE POLICY "Users can view own predictions" ON predictions 
          FOR SELECT USING (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'predictions' AND policyname = 'Users can insert own predictions') THEN
        CREATE POLICY "Users can insert own predictions" ON predictions 
          FOR INSERT WITH CHECK (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'predictions' AND policyname = 'Users can update own predictions') THEN
        CREATE POLICY "Users can update own predictions" ON predictions 
          FOR UPDATE USING (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'predictions' AND policyname = 'Users can delete own predictions') THEN
        CREATE POLICY "Users can delete own predictions" ON predictions 
          FOR DELETE USING (auth.uid() = user_id);
    END IF;
END $$;

-- Create policies for reports (only if they don't exist)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'reports' AND policyname = 'Users can view own reports') THEN
        CREATE POLICY "Users can view own reports" ON reports 
          FOR SELECT USING (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'reports' AND policyname = 'Users can insert own reports') THEN
        CREATE POLICY "Users can insert own reports" ON reports 
          FOR INSERT WITH CHECK (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'reports' AND policyname = 'Users can update own reports') THEN
        CREATE POLICY "Users can update own reports" ON reports 
          FOR UPDATE USING (auth.uid() = user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'reports' AND policyname = 'Users can delete own reports') THEN
        CREATE POLICY "Users can delete own reports" ON reports 
          FOR DELETE USING (auth.uid() = user_id);
    END IF;
END $$;

-- Create indexes for better performance
CREATE INDEX idx_cattle_data_user_id ON cattle_data(user_id);
CREATE INDEX idx_cattle_data_cattle_id ON cattle_data(cattle_id);
CREATE INDEX idx_predictions_user_id ON predictions(user_id);
CREATE INDEX idx_predictions_cattle_id ON predictions(cattle_id);
CREATE INDEX idx_predictions_type ON predictions(prediction_type);
CREATE INDEX idx_reports_user_id ON reports(user_id);
CREATE INDEX idx_reports_cattle_id ON reports(cattle_id);
```

## Installation Steps

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Set up Supabase**
   - Create a new Supabase project at https://supabase.com
   - Copy your project URL and anon key
   - Run the SQL schema above in your Supabase SQL editor
   - Configure authentication (enable email/password or other providers)

3. **Configure Environment Variables**
   - Create `.env.local` file with your Supabase credentials
   - Update the values in `lib/supabase.js` if needed

4. **Start the Development Server**
   ```bash
   npm run dev
   ```

## Features Implemented

### 1. PDF Report Generation
- **Location**: `lib/reportGenerator.js`
- **Functionality**: Generates comprehensive PDF reports with cattle information, AI predictions, and recommendations
- **Trigger**: Click "Report" button on any cattle card in the overview tab

### 2. Supabase Integration
- **Database Storage**: All predictions and reports are automatically saved to Supabase
- **User Authentication**: Reports are user-specific and secured with Row Level Security
- **Real-time Data**: Fetches latest predictions for report generation

### 3. Report Button Functionality
- **Location**: Updated `components/cattle-card.jsx`
- **Process**:
  1. Fetches latest predictions from Supabase
  2. Generates PDF report with user input and model output
  3. Saves report metadata to database
  4. Downloads PDF file automatically

### 4. Data Management
- **Removed Fake Data**: All hardcoded sample data has been removed
- **Dynamic Loading**: Data is now fetched from Supabase and AI models
- **Empty States**: Proper UI for when no data is available

## Usage Workflow

1. **Add Cattle Data**: Use the "Input Data" tab to enter cattle information
2. **Run Predictions**: Click "Predict Milk Yield" or "Detect Disease" buttons
3. **View Results**: Predictions are displayed and automatically saved to database
4. **Generate Reports**: Go to "Overview" tab and click "Report" on any cattle card
5. **Download PDF**: Report is automatically downloaded with comprehensive analysis

## SQL Queries for Common Operations

### Fetch All Reports for a User
```sql
SELECT * FROM reports 
WHERE user_id = 'user-uuid-here' 
ORDER BY created_at DESC;
```

### Get Latest Predictions for a Cattle
```sql
SELECT * FROM predictions 
WHERE user_id = 'user-uuid-here' 
AND cattle_id = 'cattle-id-here'
ORDER BY created_at DESC 
LIMIT 10;
```

### Generate Report Summary
```sql
SELECT 
  cattle_id,
  COUNT(*) as total_predictions,
  MAX(created_at) as last_prediction,
  AVG(confidence) as avg_confidence
FROM predictions 
WHERE user_id = 'user-uuid-here'
GROUP BY cattle_id;
```

## Troubleshooting

1. **PDF Generation Issues**: Ensure `jspdf` and `html2canvas` are properly installed
2. **Supabase Connection**: Verify environment variables and network connectivity
3. **Authentication**: Check if user is logged in before generating reports
4. **Model Integration**: Ensure AI models are running on localhost:8000 and localhost:8001

## Dependencies Added

- `jspdf`: PDF generation
- `html2canvas`: HTML to canvas conversion for PDF
- `@supabase/auth-helpers-nextjs`: Enhanced Supabase authentication for Next.js
