const { createClient } = require('@supabase/supabase-js')

// Read environment variables directly from .env.local
const fs = require('fs')
const path = require('path')

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

console.log('=== SUPABASE CONNECTION TEST ===')
console.log('URL:', supabaseUrl ? 'Present' : 'Missing')
console.log('Key:', supabaseAnonKey ? 'Present' : 'Missing')
console.log('Full URL:', supabaseUrl)

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Missing environment variables!')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, supabaseAnonKey)

async function testConnection() {
  try {
    // Test basic connection
    console.log('\n=== Testing basic connection ===')
    const { data, error } = await supabase.auth.getSession()
    console.log('Session test:', error ? 'Failed' : 'Success')
    if (error) console.error('Session error:', error)

    // Test database connection
    console.log('\n=== Testing database connection ===')
    const { data: tables, error: dbError } = await supabase
      .from('cattle_data')
      .select('*', { count: 'exact', head: true })
    
    console.log('Database test:', dbError ? 'Failed' : 'Success')
    if (dbError) {
      console.error('Database error:', dbError)
    } else {
      console.log('Database accessible:', tables)
    }

    // Test auth status
    console.log('\n=== Testing auth status ===')
    const { data: user } = await supabase.auth.getUser()
    console.log('Current user:', user.user ? 'Logged in' : 'Not logged in')

  } catch (error) {
    console.error('Connection test failed:', error)
  }
}

testConnection()
