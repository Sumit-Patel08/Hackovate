const { createClient } = require('@supabase/supabase-js')
const fs = require('fs')
const path = require('path')

// Read environment variables
const envPath = path.join(__dirname, '.env.local')
const envContent = fs.readFileSync(envPath, 'utf8')

let supabaseUrl, supabaseAnonKey

envContent.split('\n').forEach(line => {
  if (line.startsWith('NEXT_PUBLIC_SUPABASE_URL=')) {
    supabaseUrl = line.split('=')[1]
  }
  if (line.startsWith('NEXT_PUBLIC_SUPABASE_ANON_KEY=')) {
    supabaseAnonKey = line.split('=')[1]
  }
})

const supabase = createClient(supabaseUrl, supabaseAnonKey)

async function debugCattleSave() {
  console.log('=== CATTLE DATA SAVE DEBUG ===')
  
  try {
    // Test 1: Check authentication
    console.log('\n1. Testing authentication...')
    const { data: { user }, error: authError } = await supabase.auth.getUser()
    console.log('Current user:', user ? 'Authenticated' : 'Not authenticated')
    if (authError) console.log('Auth error:', authError)
    
    // Test 2: Check table structure
    console.log('\n2. Testing table structure...')
    const { data: tableInfo, error: tableError } = await supabase
      .from('cattle_data')
      .select('*')
      .limit(1)
    
    console.log('Table access:', tableError ? 'Failed' : 'Success')
    if (tableError) {
      console.log('Table error:', tableError)
    } else {
      console.log('Sample data structure:', tableInfo)
    }
    
    // Test 3: Test insert with minimal data (if user is authenticated)
    if (user) {
      console.log('\n3. Testing cattle data insert...')
      const testData = {
        user_id: user.id,
        cattle_id: `test-cattle-${Date.now()}`,
        breed: 'Holstein',
        age_months: 36,
        weight_kg: 550,
        feed_type: 'mixed',
        feed_quantity_kg: 15,
        grazing_hours: 6,
        body_temperature: 38.5,
        heart_rate: 60,
        environmental_data: {
          temperature: 25,
          humidity: 65,
          season: 'summer',
          housing_type: 'free_stall'
        },
        health_metrics: {
          lameness_score: 1,
          appetite_score: 4,
          coat_condition: 4,
          udder_swelling: 0,
          rumination_hours: 7,
          walking_distance_km: 3
        }
      }
      
      console.log('Attempting to insert test data:', testData)
      
      const { data: insertResult, error: insertError } = await supabase
        .from('cattle_data')
        .insert(testData)
        .select()
      
      if (insertError) {
        console.log('Insert failed!')
        console.log('Error code:', insertError.code)
        console.log('Error message:', insertError.message)
        console.log('Error details:', insertError.details)
        console.log('Error hint:', insertError.hint)
      } else {
        console.log('Insert successful!')
        console.log('Inserted data:', insertResult)
        
        // Clean up test data
        await supabase
          .from('cattle_data')
          .delete()
          .eq('cattle_id', testData.cattle_id)
        console.log('Test data cleaned up')
      }
    } else {
      console.log('\n3. Skipping insert test - user not authenticated')
      console.log('Please sign in to test cattle data insertion')
    }
    
    // Test 4: Check RLS policies
    console.log('\n4. Testing RLS policies...')
    const { data: policies, error: policyError } = await supabase.rpc('get_policies_for_table', { table_name: 'cattle_data' })
    if (policyError) {
      console.log('Could not check policies:', policyError.message)
    } else {
      console.log('RLS policies found:', policies?.length || 0)
    }
    
  } catch (error) {
    console.error('Debug test failed:', error)
  }
}

debugCattleSave()
