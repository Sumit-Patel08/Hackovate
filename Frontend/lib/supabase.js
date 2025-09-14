import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

console.log('=== SUPABASE INITIALIZATION ===')
console.log('Supabase URL:', supabaseUrl ? 'Present' : 'Missing')
console.log('Supabase Anon Key:', supabaseAnonKey ? 'Present' : 'Missing')
console.log('Full URL:', supabaseUrl)
console.log('Key length:', supabaseAnonKey?.length || 0)

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('=== MISSING SUPABASE ENVIRONMENT VARIABLES ===')
  console.error('NEXT_PUBLIC_SUPABASE_URL:', supabaseUrl)
  console.error('NEXT_PUBLIC_SUPABASE_ANON_KEY:', supabaseAnonKey ? '[REDACTED]' : 'undefined')
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Test connection
supabase.auth.getSession().then(({ data, error }) => {
  console.log('=== SUPABASE CONNECTION TEST ===')
  console.log('Session data:', data)
  console.log('Session error:', error)
})

// Database table schemas for reference:
/*
-- Cattle data table
CREATE TABLE cattle_data (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  cattle_id VARCHAR(50) NOT NULL,
  breed VARCHAR(100),
  age_months INTEGER,
  weight_kg DECIMAL(6,2),
  feed_type VARCHAR(50),
  feed_quantity_kg DECIMAL(5,2),
  grazing_hours DECIMAL(4,2),
  body_temperature DECIMAL(4,2),
  heart_rate DECIMAL(5,2),
  environmental_data JSONB,
  health_metrics JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Predictions table
CREATE TABLE predictions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
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
  user_id UUID REFERENCES auth.users(id),
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

-- Create policies
CREATE POLICY "Users can view own cattle data" ON cattle_data FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own cattle data" ON cattle_data FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own cattle data" ON cattle_data FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can view own predictions" ON predictions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own predictions" ON predictions FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own reports" ON reports FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own reports" ON reports FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own reports" ON reports FOR UPDATE USING (auth.uid() = user_id);
*/
