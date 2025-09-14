import { supabase } from './supabase'

export class CattleDataManager {
  // Save cattle data to Supabase - ALWAYS INSERT NEW RECORDS
  static async saveCattleData(userId, cattleData) {
    try {
      console.log('CattleDataManager: Attempting to save NEW cattle data for user:', userId)
      console.log('CattleDataManager: Cattle data to save:', cattleData)
      
      // Generate unique cattle_id for each new cattle
      const uniqueCattleId = `cattle-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      console.log('CattleDataManager: Generated unique cattle ID:', uniqueCattleId)
      
      // ALWAYS INSERT - Never update existing records
      console.log('CattleDataManager: Inserting NEW cattle record')
      
      // Prepare the data for insertion with proper validation
      const insertData = {
        user_id: userId,
        cattle_id: uniqueCattleId, // Always use unique ID
        breed: cattleData.breed || 'Holstein',
        age_months: Number(cattleData.age_months) || 36,
        weight_kg: Number(cattleData.weight_kg) || 550,
        feed_type: cattleData.feed_type || 'mixed',
        feed_quantity_kg: Number(cattleData.feed_quantity_kg) || 15,
        grazing_hours: Number(cattleData.grazing_hours) || 6,
        body_temperature: Number(cattleData.body_temperature) || 38.5,
        heart_rate: Number(cattleData.heart_rate) || 60,
        environmental_data: {
          temperature: Number(cattleData.temperature) || 25,
          humidity: Number(cattleData.humidity) || 65,
          season: cattleData.season || 'summer',
          housing_type: cattleData.housing_type || 'free_stall'
        },
        health_metrics: {
          lameness_score: Number(cattleData.lameness_score) || 1,
          appetite_score: Number(cattleData.appetite_score) || 4,
          coat_condition: Number(cattleData.coat_condition) || 4,
          udder_swelling: Number(cattleData.udder_swelling) || 0,
          rumination_hours: Number(cattleData.rumination_hours) || 7,
          walking_distance_km: Number(cattleData.walking_distance_km) || 3
        }
      }
      
      console.log('CattleDataManager: Prepared insert data:', insertData)
      
      const { data, error } = await supabase
        .from('cattle_data')
        .insert(insertData)
        .select()

      if (error) {
        console.error('CattleDataManager: Insert error details:', {
          message: error.message,
          details: error.details,
          hint: error.hint,
          code: error.code
        })
        throw error
      }
      
      const result = data[0]
      console.log('CattleDataManager: NEW cattle save successful:', result)
      return result
    } catch (error) {
      console.error('CattleDataManager: Error saving cattle data:', error)
      throw error
    }
  }

  // Fetch ALL cattle data for a user (not just most recent)
  static async fetchUserCattleData(userId) {
    try {
      console.log('=== FETCHING ALL CATTLE DATA ===')
      console.log('CattleDataManager: User ID:', userId)
      console.log('CattleDataManager: User ID type:', typeof userId)
      console.log('CattleDataManager: Supabase client:', !!supabase)
      
      // First, let's test a simple count query
      const { count, error: countError } = await supabase
        .from('cattle_data')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', userId)
      
      console.log('CattleDataManager: Total count for user:', count)
      console.log('CattleDataManager: Count error:', countError)
      
      // Fetch ALL cattle data for the user (remove any limits)
      const { data, error } = await supabase
        .from('cattle_data')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })

      console.log('CattleDataManager: Raw Supabase response - ALL data:', data)
      console.log('CattleDataManager: Raw Supabase response - error:', error)
      console.log('CattleDataManager: Total records found:', data?.length || 0)
      
      // Log each cattle record for debugging
      if (data && data.length > 0) {
        data.forEach((cattle, index) => {
          console.log(`CattleDataManager: Cattle ${index + 1}:`, {
            id: cattle.id,
            cattle_id: cattle.cattle_id,
            breed: cattle.breed,
            age: cattle.age,
            created_at: cattle.created_at
          })
        })
      }
      
      // Test without user filter to see if data exists at all
      const { data: allData, error: allError } = await supabase
        .from('cattle_data')
        .select('*')
        .limit(10)
      
      console.log('CattleDataManager: All cattle data (no filter):', allData)
      console.log('CattleDataManager: All data error:', allError)

      if (error) {
        console.error('CattleDataManager: Supabase error details:', {
          message: error.message,
          details: error.details,
          hint: error.hint,
          code: error.code
        })
        throw error
      }

      console.log('CattleDataManager: Successfully fetched ALL cattle data:', data)
      return data || []
    } catch (error) {
      console.error('CattleDataManager: Exception in fetchUserCattleData:', error)
      throw error
    }
  }

  // Get cattle data with latest predictions
  static async getCattleWithPredictions(userId) {
    try {
      console.log('CattleDataManager: Getting cattle with predictions for user:', userId)
      const cattleData = await this.fetchUserCattleData(userId)
      console.log('CattleDataManager: Raw cattle data:', cattleData)
      console.log('CattleDataManager: Raw cattle data count:', cattleData?.length || 0)
      
      if (!cattleData || cattleData.length === 0) {
        console.log('CattleDataManager: No cattle data found - returning empty array')
        return []
      }
      
      console.log('CattleDataManager: Processing', cattleData.length, 'cattle records')

      const cattleWithPredictions = await Promise.all(cattleData.map(async (cattle) => {
        console.log('CattleDataManager: Processing cattle:', cattle.cattle_id)
        
        // Fetch latest predictions from database for this cattle
        const { data: milkPredictions } = await supabase
          .from('predictions')
          .select('*')
          .eq('user_id', userId)
          .eq('cattle_id', cattle.cattle_id)
          .eq('prediction_type', 'milk_yield')
          .order('created_at', { ascending: false })
          .limit(1)

        const { data: diseasePredictions } = await supabase
          .from('predictions')
          .select('*')
          .eq('user_id', userId)
          .eq('cattle_id', cattle.cattle_id)
          .eq('prediction_type', 'disease')
          .order('created_at', { ascending: false })
          .limit(1)

        const latestMilkPrediction = milkPredictions?.[0] || null
        const latestDiseasePrediction = diseasePredictions?.[0] || null

        // Use AI model predictions if available, otherwise use default values
        let dailyYield = 20.0
        let healthScore = 75

        if (latestMilkPrediction?.prediction_result?.predicted_milk_yield) {
          dailyYield = parseFloat(latestMilkPrediction.prediction_result.predicted_milk_yield)
        }

        if (latestDiseasePrediction?.prediction_result) {
          // Calculate health score based on disease prediction
          const diseaseResult = latestDiseasePrediction.prediction_result
          if (diseaseResult.predicted_disease === 'healthy') {
            healthScore = 90
          } else {
            const confidence = diseaseResult.confidence || 0
            const riskLevel = diseaseResult.risk_level
            
            if (riskLevel === 'high') {
              healthScore = Math.max(30, 70 - (confidence * 40))
            } else if (riskLevel === 'medium') {
              healthScore = Math.max(50, 80 - (confidence * 30))
            } else {
              healthScore = Math.max(70, 85 - (confidence * 15))
            }
          }
        }

        return {
          ...cattle,
          dailyYield: parseFloat(dailyYield.toFixed(1)),
          healthScore: Math.round(healthScore),
          milkPrediction: latestMilkPrediction,
          diseasePrediction: latestDiseasePrediction,
          lastPredictionDate: latestMilkPrediction?.created_at || latestDiseasePrediction?.created_at || cattle.created_at,
          daysAgo: this.getDaysAgo(latestMilkPrediction?.created_at || latestDiseasePrediction?.created_at || cattle.created_at)
        }
      }))

      console.log('CattleDataManager: Final cattle with predictions:', cattleWithPredictions)
      return cattleWithPredictions
    } catch (error) {
      console.error('CattleDataManager: Error in getCattleWithPredictions:', error)
      return []
    }
  }

  // Estimate daily yield based on cattle characteristics
  static estimateDailyYield(cattle) {
    let baseYield = 20.0 // Base yield for average cattle
    
    // Adjust based on breed
    const breedMultipliers = {
      'Holstein': 1.3,
      'Jersey': 0.9,
      'Gir': 0.7,
      'Sahiwal': 0.8,
      'Red Sindhi': 0.75,
      'Crossbred': 1.1
    }
    
    const breedMultiplier = breedMultipliers[cattle.breed] || 1.0
    baseYield *= breedMultiplier
    
    // Adjust based on age (peak production around 3-6 years)
    const ageMonths = cattle.age_months || 36
    if (ageMonths < 24) baseYield *= 0.7 // Young cattle
    else if (ageMonths > 96) baseYield *= 0.8 // Older cattle
    
    // Adjust based on weight
    const weight = cattle.weight_kg || 500
    if (weight > 600) baseYield *= 1.1
    else if (weight < 400) baseYield *= 0.9
    
    // Adjust based on feed quality and quantity
    const feedQuantity = cattle.feed_quantity_kg || 15
    if (feedQuantity > 20) baseYield *= 1.1
    else if (feedQuantity < 10) baseYield *= 0.8
    
    // Adjust based on health metrics
    if (cattle.health_metrics) {
      const health = cattle.health_metrics
      if (health.appetite_score < 3) baseYield *= 0.8
      if (health.lameness_score > 2) baseYield *= 0.9
      if (health.udder_swelling > 1) baseYield *= 0.7
    }
    
    return Math.max(baseYield, 5.0) // Minimum 5L per day
  }

  // Calculate health score based on various factors
  static calculateHealthScore(cattle, diseasePrediction) {
    let score = 100

    // Reduce score based on health metrics
    if (cattle.health_metrics?.lameness_score > 2) score -= 20
    if (cattle.health_metrics?.appetite_score < 3) score -= 15
    if (cattle.health_metrics?.coat_condition < 3) score -= 10
    if (cattle.health_metrics?.udder_swelling > 1) score -= 25

    // Factor in disease prediction
    if (diseasePrediction?.prediction_result?.predicted_disease !== 'healthy') {
      const riskLevel = diseasePrediction.prediction_result.risk_level
      if (riskLevel === 'high') score -= 30
      else if (riskLevel === 'medium') score -= 15
      else if (riskLevel === 'low') score -= 5
    }

    return Math.max(score, 0)
  }

  // Calculate days ago from timestamp
  static getDaysAgo(timestamp) {
    const now = new Date()
    const past = new Date(timestamp)
    const diffTime = Math.abs(now - past)
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
    return diffDays
  }

  // Get dashboard statistics
  static async getDashboardStats(userId) {
    try {
      const cattleData = await this.getCattleWithPredictions(userId)
      
      const totalCattle = cattleData.length
      const avgYield = totalCattle > 0 
        ? (cattleData.reduce((sum, cattle) => sum + cattle.dailyYield, 0) / totalCattle).toFixed(1)
        : "0.0"
      const healthyAnimals = cattleData.filter(cattle => cattle.healthScore >= 80).length

      // Get recent predictions for chart data
      const { data: recentPredictions } = await supabase
        .from('predictions')
        .select('*')
        .eq('user_id', userId)
        .eq('prediction_type', 'milk_yield')
        .order('created_at', { ascending: false })
        .limit(7)

      const milkYieldData = recentPredictions?.map((pred, index) => ({
        day: new Date(pred.created_at).toLocaleDateString('en-US', { weekday: 'short' }),
        yield: pred.prediction_result?.predicted_milk_yield || 0,
        predicted: pred.prediction_result?.predicted_milk_yield || 0
      })) || []

      return {
        totalCattle,
        avgYield,
        healthyAnimals,
        milkYieldData: milkYieldData.reverse() // Show chronologically
      }
    } catch (error) {
      console.error('Error getting dashboard stats:', error)
      return {
        totalCattle: 0,
        avgYield: "0.0",
        healthyAnimals: 0,
        milkYieldData: []
      }
    }
  }
}
