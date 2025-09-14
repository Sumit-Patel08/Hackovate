-- Add new columns to existing cattle_data table
ALTER TABLE cattle_data 
ADD COLUMN IF NOT EXISTS feed_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS feed_quantity_kg DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS grazing_hours DECIMAL(4,2),
ADD COLUMN IF NOT EXISTS body_temperature DECIMAL(4,2),
ADD COLUMN IF NOT EXISTS heart_rate DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS environmental_data JSONB,
ADD COLUMN IF NOT EXISTS health_metrics JSONB;

-- Enable Row Level Security
ALTER TABLE cattle_data ENABLE ROW LEVEL SECURITY;

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

-- Create indexes for better performance (only if they don't exist)
CREATE INDEX IF NOT EXISTS idx_cattle_data_user_id ON cattle_data(user_id);
CREATE INDEX IF NOT EXISTS idx_cattle_data_cattle_id ON cattle_data(cattle_id);

-- Update function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger only if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_cattle_data_updated_at') THEN
        CREATE TRIGGER update_cattle_data_updated_at 
        BEFORE UPDATE ON cattle_data 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;
