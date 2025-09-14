"use client"

import { useState, useEffect } from "react"
import { useAuth } from "@/contexts/AuthContext"
import { supabase } from "@/lib/supabase"
import { savePredictionToDatabase, fetchCattleData, ReportGenerator } from "@/lib/reportGenerator"
import { CattleDataManager } from "@/lib/cattleDataManager"
import { getTranslation } from "@/lib/translations"
import UserProfile from "@/components/UserProfile"
import ProtectedRoute from "@/components/ProtectedRoute"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import {
  Milk,
  Heart,
  RefreshCw,
  FileText,
  Cog as Cow,
  Activity,
  AlertTriangle,
  CheckCircle,
  Download,
  Globe,
  Plus,
} from "lucide-react"
import { CattleCard } from "@/components/cattle-card"

// Real-time data will be fetched from Supabase and AI predictions

export default function DairyDashboard() {
  const { user } = useAuth()
  const [language, setLanguage] = useState("en")
  
  // Create translation helper function
  const t = (key) => getTranslation(language, key)
  const [cattleData, setCattleData] = useState([])
  const [milkYieldData, setMilkYieldData] = useState([])
  const [formData, setFormData] = useState({
    // Animal-related data
    breed: "Holstein",
    age_months: 36,
    weight_kg: 550,
    lactation_stage: "peak",
    lactation_day: 150,
    parity: 2,
    historical_yield_7d: 25.0,
    historical_yield_30d: 24.0,
    
    // Feed and nutrition data
    feed_type: "mixed",
    feed_quantity_kg: 15,
    feeding_frequency: 3,
    
    // Activity & behavioral data
    walking_distance_km: 3.0,
    grazing_hours: 6.0,
    rumination_hours: 7.0,
    resting_hours: 11.0,
    
    // Health data
    body_temperature: 38.5,
    heart_rate: 60.0,
    health_score: 0.9,
    
    // Comprehensive health parameters
    white_blood_cells: 7500,
    somatic_cell_count: 150000,
    rumen_ph: 6.3,
    rumen_temperature: 40.0,
    calcium_level: 10.0,
    phosphorus_level: 5.0,
    protein_level: 7.0,
    glucose_level: 60,
    udder_swelling: 0,
    lameness_score: 1,
    appetite_score: 4,
    coat_condition: 4,
    
    // Environmental data
    temperature: 25,
    humidity: 65,
    season: "summer",
    housing_type: "free_stall",
    ventilation_score: 0.8,
    cleanliness_score: 0.8,
    day_of_year: 180,
  })

  const [milkPrediction, setMilkPrediction] = useState(null)
  const [diseasePrediction, setDiseasePrediction] = useState(null)
  const [loadingMilk, setLoadingMilk] = useState(false)
  const [loadingDisease, setLoadingDisease] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState("input")
  const [predictionType, setPredictionType] = useState("milk") // "milk" or "disease"
  const [validationError, setValidationError] = useState("")
  const [showValidationDialog, setShowValidationDialog] = useState(false)
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [savedCattleId, setSavedCattleId] = useState(null)
  const [isSavingData, setIsSavingData] = useState(false)

  // Use the translation helper function
  // const t = translations[language] // Remove this line

  // Load cattle data when component mounts or user changes
  useEffect(() => {
    if (user) {
      console.log('=== USER LOGIN DETECTED ===')
      console.log('User object:', user)
      console.log('User ID:', user.id)
      console.log('User email:', user.email)
      
      loadCattleData()
      checkPredictionsData()
      loadLastCattleFormData()
      
      // Set up auto-refresh interval
      const interval = setInterval(() => {
        if (document.visibilityState === 'visible') {
          loadCattleData()
        }
      }, 30000) // Refresh every 30 seconds when page is visible
      
      return () => clearInterval(interval)
    } else {
      console.log('=== NO USER DETECTED ===')
      setCattleData([])
      setMilkYieldData([])
      setMilkPrediction(null)
      setDiseasePrediction(null)
      setSavedCattleId(null)
    }
  }, [user])

  // Function to check predictions table data
  const checkPredictionsData = async () => {
    try {
      const { data: predictions, error } = await supabase
        .from('predictions')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(10)
      
      if (error) {
        console.error('Error fetching predictions:', error)
        return
      }
      
      console.log('=== PREDICTIONS TABLE DATA ===')
      console.log('Predictions table data:', predictions)
      console.log('Total predictions found:', predictions?.length || 0)
      
      if (predictions && predictions.length > 0) {
        predictions.forEach((pred, index) => {
          console.log(`Prediction ${index + 1}:`, {
            id: pred.id,
            cattle_id: pred.cattle_id,
            prediction_type: pred.prediction_type,
            prediction_result: pred.prediction_result,
            input_data: pred.input_data,
            confidence: pred.confidence,
            created_at: pred.created_at
          })
        })
      } else {
        console.log('No predictions found in database for user:', user.id)
      }
    } catch (error) {
      console.error('Error checking predictions data:', error)
    }
  }

  const loadCattleData = async () => {
    if (!user) {
      console.log('loadCattleData: No user found')
      return
    }
    
    try {
      console.log('=== LOAD CATTLE DATA START ===')
      console.log('loadCattleData: Loading cattle data for user:', user.id)
      console.log('loadCattleData: User object:', user)
      console.log('loadCattleData: User email:', user.email)
      
      // Test basic Supabase connection with detailed logging
      console.log('Testing basic Supabase connection...')
      const { data: testData, error: testError } = await supabase
        .from('cattle_data')
        .select('id, user_id, cattle_id, created_at')
        .eq('user_id', user.id)
      
      console.log('loadCattleData: Supabase test query result:', testData)
      console.log('loadCattleData: Supabase test error:', testError)
      console.log('loadCattleData: Test data count:', testData?.length || 0)
      
      // Also test without user filter
      const { data: allTestData, error: allTestError } = await supabase
        .from('cattle_data')
        .select('id, user_id, cattle_id, created_at')
        .limit(10)
      
      console.log('loadCattleData: All data (no filter):', allTestData)
      console.log('loadCattleData: All data error:', allTestError)
      
      console.log('Calling CattleDataManager.getCattleWithPredictions...')
      const data = await CattleDataManager.getCattleWithPredictions(user.id)
      console.log('loadCattleData: Loaded cattle data from manager:', data)
      console.log('loadCattleData: Cattle data length:', data?.length || 0)
      
      // Force state update with fresh data
      console.log('Setting cattleData state with:', data?.length || 0, 'records')
      setCattleData(data || [])
      console.log('State updated - cattleData should now have:', data?.length || 0, 'records')
      
      console.log('Calling CattleDataManager.getDashboardStats...')
      const stats = await CattleDataManager.getDashboardStats(user.id)
      console.log('loadCattleData: Loaded dashboard stats:', stats)
      setMilkYieldData(stats.milkYieldData)
      
      console.log('=== LOAD CATTLE DATA COMPLETE ===')
    } catch (error) {
      console.error('=== LOAD CATTLE DATA ERROR ===')
      console.error('loadCattleData: Error loading cattle data:', error)
      console.error('Error details:', error.message, error.details)
      setCattleData([])
      setMilkYieldData([])
    }
  }

  const testDirectQuery = async () => {
    if (!user) {
      console.log('No user for direct query test')
      return
    }

    try {
      console.log('=== DIRECT QUERY TEST ===')
      console.log('Testing direct Supabase query for user:', user.id)
      
      // Test basic connection
      const { data: testData, error: testError, count } = await supabase
        .from('cattle_data')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', user.id)
      
      console.log('Direct query - count result:', count)
      console.log('Direct query - count error:', testError)
      
      // Test full data fetch
      const { data: fullData, error: fullError } = await supabase
        .from('cattle_data')
        .select('*')
        .eq('user_id', user.id)
      
      console.log('Direct query - full data:', fullData)
      console.log('Direct query - full error:', fullError)
      
      // Test auth session
      const { data: session, error: sessionError } = await supabase.auth.getSession()
      console.log('Auth session:', session)
      console.log('Auth session error:', sessionError)
      
      alert(`Direct Query Results:\nCount: ${count || 0}\nFull Data: ${fullData?.length || 0} records\nErrors: ${testError?.message || fullError?.message || 'None'}`)
      
    } catch (error) {
      console.error('Direct query test failed:', error)
      alert(`Direct Query Failed: ${error.message}`)
    }
  }

  const loadLastCattleFormData = async () => {
    if (!user) return
    
    try {
      console.log('Loading last cattle form data for user:', user.id)
      
      // Get the most recent cattle entry
      const { data: recentCattle, error } = await supabase
        .from('cattle_data')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(1)
      
      if (error) {
        console.error('Error loading recent cattle data:', error)
        return
      }
      
      if (recentCattle && recentCattle.length > 0) {
        const cattle = recentCattle[0]
        console.log('Loading form data from recent cattle:', cattle)
        
        // Update form with the most recent cattle data
        setFormData({
          ...formData,
          breed: cattle.breed || formData.breed,
          age_months: cattle.age_months || formData.age_months,
          weight_kg: cattle.weight_kg || formData.weight_kg,
          feed_type: cattle.feed_type || formData.feed_type,
          feed_quantity_kg: cattle.feed_quantity_kg || formData.feed_quantity_kg,
          grazing_hours: cattle.grazing_hours || formData.grazing_hours,
          body_temperature: cattle.body_temperature || formData.body_temperature,
          heart_rate: cattle.heart_rate || formData.heart_rate,
          temperature: cattle.environmental_data?.temperature || formData.temperature,
          humidity: cattle.environmental_data?.humidity || formData.humidity,
          season: cattle.environmental_data?.season || formData.season,
          housing_type: cattle.environmental_data?.housing_type || formData.housing_type,
          lameness_score: cattle.health_metrics?.lameness_score || formData.lameness_score,
          appetite_score: cattle.health_metrics?.appetite_score || formData.appetite_score,
          coat_condition: cattle.health_metrics?.coat_condition || formData.coat_condition,
          udder_swelling: cattle.health_metrics?.udder_swelling || formData.udder_swelling,
          rumination_hours: cattle.health_metrics?.rumination_hours || formData.rumination_hours,
          walking_distance_km: cattle.health_metrics?.walking_distance_km || formData.walking_distance_km
        })
        
        setSavedCattleId(cattle.cattle_id)
        
        // Load recent predictions for this cattle
        await loadRecentPredictions(cattle.cattle_id)
      }
    } catch (error) {
      console.error('Error in loadLastCattleFormData:', error)
    }
  }

  const loadRecentPredictions = async (cattleId) => {
    if (!user || !cattleId) return
    
    try {
      console.log('Loading recent predictions for cattle:', cattleId)
      
      // Get latest milk prediction
      const { data: milkPredictions } = await supabase
        .from('predictions')
        .select('*')
        .eq('user_id', user.id)
        .eq('cattle_id', cattleId)
        .eq('prediction_type', 'milk_yield')
        .order('created_at', { ascending: false })
        .limit(1)
      
      if (milkPredictions && milkPredictions.length > 0) {
        console.log('Loading milk prediction:', milkPredictions[0])
        setMilkPrediction(milkPredictions[0].prediction_result)
      }
      
      // Get latest disease prediction
      const { data: diseasePredictions } = await supabase
        .from('predictions')
        .select('*')
        .eq('user_id', user.id)
        .eq('cattle_id', cattleId)
        .eq('prediction_type', 'disease_detection')
        .order('created_at', { ascending: false })
        .limit(1)
      
      if (diseasePredictions && diseasePredictions.length > 0) {
        console.log('Loading disease prediction:', diseasePredictions[0])
        setDiseasePrediction(diseasePredictions[0].prediction_result)
      }
    } catch (error) {
      console.error('Error loading recent predictions:', error)
    }
  }

  const validateAllInputs = () => {
    const errors = []
    
    // Validate age_months
    if (formData.age_months < 24 || formData.age_months > 120) {
      errors.push("Age must be between 24-120 months")
    }
    
    // Validate weight_kg
    if (formData.weight_kg < 300 || formData.weight_kg > 1200) {
      errors.push("Weight must be between 300-1200 kg")
    }
    
    // Validate feed_quantity_kg
    if (formData.feed_quantity_kg < 5 || formData.feed_quantity_kg > 50) {
      errors.push("Feed quantity must be between 5-50 kg per day")
    }
    
    // Validate body temperature
    if (formData.body_temperature < 36 || formData.body_temperature > 45) {
      errors.push("Body temperature must be between 36°C to 45°C")
    }
    
    // Validate humidity
    if (formData.humidity < 0 || formData.humidity > 100) {
      errors.push("Humidity must be between 0-100%")
    }
    
    if (errors.length > 0) {
      setValidationError("Invalid input: " + errors.join(", "))
      setShowValidationDialog(true)
      return false
    }
    
    setValidationError("")
    return true
  }

  const handleInputChange = (field, value) => {
    // Always allow the input to be set
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleMilkPrediction = async () => {
    // Validate inputs first
    if (!validateAllInputs()) {
      return
    }

    setLoadingMilk(true)
    setError(null)
    setMilkPrediction(null)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      const data = await response.json()

      if (response.ok) {
        setMilkPrediction(data)
        
        // Save prediction to Supabase
        if (user) {
          try {
            const cattleId = savedCattleId || `cattle-${Date.now()}`
            await savePredictionToDatabase(
              user.id,
              cattleId,
              'milk_yield',
              formData,
              data
            )
            
            // Show save dialog if cattle data not saved yet
            if (!savedCattleId) {
              setShowSaveDialog(true)
            }
            
            // Refresh data immediately after prediction
            setTimeout(() => loadCattleData(), 500)
          } catch (dbError) {
            console.error('Error saving milk prediction to database:', dbError)
          }
        }
      } else {
        setError(data.detail || 'Milk yield prediction failed')
      }
    } catch (err) {
      setError('Failed to connect to milk yield service. Make sure Model 1 is running on http://localhost:8000')
    } finally {
      setLoadingMilk(false)
    }
  }

  const handleDiseasePrediction = async () => {
    // Validate inputs first
    if (!validateAllInputs()) {
      return
    }

    setLoadingDisease(true)
    setError(null)
    setDiseasePrediction(null)

    // Prepare data for disease detection (add required fields)
    const diseaseData = {
      ...formData,
      // Add disease-specific fields with defaults if missing
      white_blood_cells: formData.white_blood_cells || 7500,
      somatic_cell_count: formData.somatic_cell_count || 150000,
      rumen_ph: formData.rumen_ph || 6.3,
      rumen_temperature: formData.rumen_temperature || 40.0,
      calcium_level: formData.calcium_level || 10.0,
      phosphorus_level: formData.phosphorus_level || 5.0,
      protein_level: formData.protein_level || 7.0,
      glucose_level: formData.glucose_level || 60,
      udder_swelling: formData.udder_swelling || 0,
      lameness_score: formData.lameness_score || 1,
      appetite_score: formData.appetite_score || 4,
      coat_condition: formData.coat_condition || 4,
      respiratory_rate: 28
    }

    try {
      const response = await fetch('http://localhost:8001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(diseaseData),
      })

      const data = await response.json()

      if (response.ok) {
        setDiseasePrediction(data)
        
        // Save prediction to Supabase
        if (user) {
          try {
            const cattleId = savedCattleId || `cattle-${Date.now()}`
            await savePredictionToDatabase(
              user.id,
              cattleId,
              'disease_detection',
              formData,
              data
            )
            
            // Show save dialog if cattle data not saved yet
            if (!savedCattleId) {
              setShowSaveDialog(true)
            }
            
            // Refresh data immediately after prediction
            setTimeout(() => loadCattleData(), 500)
          } catch (dbError) {
            console.error('Error saving disease prediction to database:', dbError)
          }
        }
      } else {
        setError(data.detail || 'Disease detection failed')
      }
    } catch (err) {
      setError('Failed to connect to disease detection service. Make sure Model 2 is running on http://localhost:8001')
    } finally {
      setLoadingDisease(false)
    }
  }

  const handleBothPredictions = async () => {
    await Promise.all([handleMilkPrediction(), handleDiseasePrediction()])
  }

  const handleSaveCattleData = async () => {
    console.log('=== SAVE CATTLE DATA ATTEMPT ===')
    console.log('User object:', user)
    console.log('User ID:', user?.id)
    console.log('User authenticated:', !!user)
    
    if (!user) {
      console.error('No user found - authentication required')
      alert('Please log in to save cattle data')
      return
    }

    // Validate inputs before saving
    console.log('Validating inputs...')
    if (!validateAllInputs()) {
      console.error('Input validation failed')
      return
    }
    console.log('Input validation passed')

    setIsSavingData(true)
    
    try {
      const cattleId = savedCattleId || `cattle-${Date.now()}`
      const cattleDataToSave = {
        ...formData,
        cattle_id: cattleId
      }
      
      console.log('=== ATTEMPTING TO SAVE ===')
      console.log('User ID for save:', user.id)
      console.log('Cattle data to save:', cattleDataToSave)
      console.log('Form data keys:', Object.keys(formData))
      console.log('Required fields check:', {
        breed: formData.breed,
        age_months: formData.age_months,
        weight_kg: formData.weight_kg,
        body_temperature: formData.body_temperature
      })
      
      const savedData = await CattleDataManager.saveCattleData(user.id, cattleDataToSave)
      console.log('=== SAVE SUCCESSFUL ===')
      console.log('Saved cattle data result:', savedData)
      
      setSavedCattleId(savedData.cattle_id)
      setShowSaveDialog(false)
      
      // Force reload cattle data immediately after save
      console.log('=== RELOADING DATA AFTER SAVE ===')
      await loadCattleData()
      console.log('Data reloaded after save - cattleData length:', cattleData.length)
      
      alert('Cattle data saved successfully!')
    } catch (error) {
      console.error('=== SAVE FAILED ===')
      console.error('Error saving cattle data:', error)
      console.error('Error message:', error.message)
      console.error('Error details:', error.details)
      console.error('Error code:', error.code)
      console.error('Error hint:', error.hint)
      console.error('Full error object:', JSON.stringify(error, null, 2))
      alert(`Failed to save cattle data: ${error.message || 'Unknown error'}. Check browser console for full details.`)
    } finally {
      setIsSavingData(false)
    }
  }

  const handleReloadData = async () => {
    console.log('=== MANUAL RELOAD DATA TRIGGERED ===')
    setError(null)
    
    try {
      await loadCattleData()
      alert('Cattle data reloaded successfully!')
    } catch (error) {
      console.error('Error reloading data:', error)
      alert('Failed to reload data. Please check console for details.')
    }
  }

  const handleGenerateDashboardReport = async () => {
    if (!user) {
      alert('Please log in to generate reports')
      return
    }

    if (cattleData.length === 0) {
      alert('No cattle data available. Please add some cattle and run predictions first.')
      return
    }

    try {
      console.log('Starting report generation...')
      console.log('Cattle data for report:', cattleData)
      console.log('User ID:', user.id)
      console.log('Language:', language)
      
      const reportGenerator = new ReportGenerator(language)
      console.log('ReportGenerator created successfully')
      
      const doc = await reportGenerator.generateDashboardReport(cattleData, user.id)
      console.log('Dashboard report generated successfully')
      
      // Generate filename with current date
      const currentDate = new Date().toISOString().split('T')[0]
      const filename = `farm-dashboard-report-${currentDate}.pdf`
      console.log('Generated filename:', filename)
      
      // Download the PDF
      reportGenerator.downloadPDF(filename)
      console.log('PDF download initiated')
      
      // Save report metadata to database (optional - don't fail if this errors)
      try {
        const reportData = {
          total_cattle: cattleData.length,
          total_daily_yield: cattleData.reduce((sum, cattle) => sum + (cattle.dailyYield || 0), 0),
          healthy_cattle: cattleData.filter(cattle => (cattle.healthScore || 0) >= 80).length,
          report_type: 'dashboard_summary',
          generated_at: new Date().toISOString()
        }
        
        await reportGenerator.saveReportToDatabase(user.id, 'dashboard', reportData, reportGenerator.getPDFBlob())
        console.log('Report metadata saved to database')
      } catch (dbError) {
        console.error('Error saving report metadata (non-critical):', dbError)
        // Don't show error to user as the PDF was still generated successfully
      }
      
      alert(`Dashboard report generated successfully! Downloaded as ${filename}`)
    } catch (error) {
      console.error('=== REPORT GENERATION ERROR ===')
      console.error('Full error object:', error)
      console.error('Error message:', error.message)
      console.error('Error stack:', error.stack)
      console.error('Cattle data length:', cattleData?.length || 0)
      console.error('User object:', user)
      
      // More specific error messages
      if (error.message?.includes('jsPDF')) {
        alert('PDF generation library error. Please refresh the page and try again.')
      } else if (error.message?.includes('No cattle data')) {
        alert('No cattle data available for report. Please add cattle data first.')
      } else if (error.message?.includes('translations')) {
        alert('Translation error. Trying with default language...')
        // Retry with English
        try {
          const reportGenerator = new ReportGenerator('en')
          const doc = await reportGenerator.generateDashboardReport(cattleData, user.id)
          const filename = `farm-dashboard-report-${new Date().toISOString().split('T')[0]}.pdf`
          reportGenerator.downloadPDF(filename)
          alert(`Report generated successfully in English! Downloaded as ${filename}`)
        } catch (retryError) {
          console.error('Retry with English also failed:', retryError)
          alert('Failed to generate report. Please check console for details.')
        }
      } else {
        alert(`Failed to generate report: ${error.message}. Please check console for details.`)
      }
    }
  }

  const getHealthStatusColor = (status) => {
    switch (status) {
      case "normal":
        return "bg-green-100 text-green-800"
      case "low":
        return "bg-yellow-100 text-yellow-800"
      case "high":
        return "bg-red-100 text-red-800"
      case "good":
        return "bg-blue-100 text-blue-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const totalCattle = cattleData.length
  const avgYield = totalCattle > 0 ? (cattleData.reduce((sum, cattle) => sum + (cattle.dailyYield || 0), 0) / totalCattle).toFixed(1) : "0.0"
  
  // Enhanced health score calculation with fallback logic
  const healthyAnimals = cattleData.filter((cattle) => {
    console.log('Processing cattle for health:', {
      cattle_id: cattle.cattle_id,
      healthScore: cattle.healthScore,
      health_metrics: cattle.health_metrics,
      diseasePrediction: cattle.diseasePrediction
    })
    
    // Primary: Use calculated healthScore from predictions
    if (cattle.healthScore !== undefined && cattle.healthScore !== null) {
      console.log(`Using healthScore: ${cattle.healthScore} >= 80 = ${cattle.healthScore >= 80}`)
      return cattle.healthScore >= 80
    }
    
    // Fallback: Calculate health score from available data
    let calculatedScore = 85 // Default healthy score
    
    // Check health metrics if available
    if (cattle.health_metrics) {
      const metrics = cattle.health_metrics
      if (metrics.lameness_score > 2) calculatedScore -= 20
      if (metrics.appetite_score < 3) calculatedScore -= 15
      if (metrics.coat_condition < 3) calculatedScore -= 10
      if (metrics.udder_swelling > 1) calculatedScore -= 25
    }
    
    // Check disease prediction if available
    if (cattle.diseasePrediction?.prediction_result?.predicted_disease !== 'healthy') {
      calculatedScore -= 20
    }
    
    // If no specific health data, assume healthy if cattle exists
    if (!cattle.health_metrics && !cattle.diseasePrediction) {
      calculatedScore = 85 // Default to healthy
    }
    
    console.log(`Calculated health score: ${calculatedScore} >= 80 = ${calculatedScore >= 80}`)
    return calculatedScore >= 80
  }).length
  
  console.log('Dashboard Health Calculation Summary:', {
    totalCattle,
    cattleDataLength: cattleData.length,
    cattleIds: cattleData.map(c => c.cattle_id),
    healthyAnimals,
    healthPercentage: cattleData.length > 0 ? Math.round((healthyAnimals / cattleData.length) * 100) : 0
  })

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {/* User Profile */}
              <UserProfile />
              
              <div className="flex items-center space-x-3">
                <Cow className="h-8 w-8 text-primary" />
                <div>
                  <h1 className="text-2xl font-bold text-foreground">{t('title')}</h1>
                  <p className="text-sm text-muted-foreground">{t('subtitle')}</p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Language Selector */}
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="w-32">
                  <Globe className="h-4 w-4 mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="en">🇬🇧 English</SelectItem>
                  <SelectItem value="hi">🇮🇳 हिंदी</SelectItem>
                  <SelectItem value="gu">🇮🇳 ગુજરાતી</SelectItem>
                  <SelectItem value="mr">🇮🇳 मराठी</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="input">{t('inputData')}</TabsTrigger>
            <TabsTrigger value="dashboard">{t('dashboard')}</TabsTrigger>
            <TabsTrigger value="overview">{t('overview')}</TabsTrigger>
          </TabsList>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">{t('dashboard')}</h1>
                <p className="text-muted-foreground">Monitor your cattle health and milk production with AI-powered insights</p>
              </div>
            </div>
            {/* Debug Panel */}
            {user && (
              <div className="mb-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-gray-800">Debug Info</span>
                  <div className="flex space-x-2">
                    <Button 
                      onClick={loadCattleData}
                      size="sm"
                      variant="outline"
                    >
                      Reload Data
                    </Button>
                    <Button 
                      onClick={testDirectQuery}
                      size="sm"
                      variant="outline"
                    >
                      Test Query
                    </Button>
                  </div>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>User ID: {user.id}</div>
                  <div>User Email: {user.email}</div>
                  <div>Cattle Data Count: {cattleData.length}</div>
                  <div>Milk Yield Data Count: {milkYieldData.length}</div>
                  <div>Environment URL: {process.env.NEXT_PUBLIC_SUPABASE_URL ? 'Set' : 'Missing'}</div>
                  <div>Environment Key: {process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ? 'Set' : 'Missing'}</div>
                </div>
              </div>
            )}

            {/* Welcome Message for Returning Users */}
            {user && cattleData.length > 0 && (
              <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-blue-600" />
                  <span className="font-semibold text-blue-800">Welcome back!</span>
                </div>
                <p className="text-blue-700 text-sm mt-1">
                  Loaded {cattleData.length} cattle records from your previous sessions. 
                  Your most recent data has been restored to the Input tab.
                </p>
              </div>
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{t('totalCattle')}</CardTitle>
                  <Cow className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{cattleData.length}</div>
                  <p className="text-xs text-muted-foreground">
                    {cattleData.length > 0 ? 'Active cattle in system' : 'No cattle data yet'}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{t('avgYield')}</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {cattleData.length > 0 
                      ? (cattleData.reduce((sum, cattle) => sum + (cattle.dailyYield || 0), 0) / cattleData.length).toFixed(1)
                      : "0.0"
                    } L
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {cattleData.length > 0 ? 'Average daily production' : 'No predictions yet'}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{t('healthyAnimals')}</CardTitle>
                  <Heart className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {healthyAnimals}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {cattleData.length > 0 
                      ? `${healthyAnimals} out of ${cattleData.length} cows healthy`
                      : "No cattle data available"
                    }
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Main Dashboard Cards */}
            <div className="grid gap-6 md:grid-cols-1">
              {/* Milk Yield Prediction Card */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Milk className="h-5 w-5 text-primary" />
                    <span>{t('milkYieldPrediction')}</span>
                  </CardTitle>
                  <CardDescription>
                    {t('predictedYield')}: <span className="font-semibold text-primary">{avgYield} {t('litersPerDay')}</span>
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="text-sm font-medium text-muted-foreground">{t('weeklyTrend')}</div>
                    {milkYieldData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={milkYieldData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="day" />
                          <YAxis />
                          <Tooltip />
                          <Line
                            type="monotone"
                            dataKey="yield"
                            stroke="#2563eb"
                            strokeWidth={2}
                            name={t('actualYield')}
                          />
                          <Line
                            type="monotone"
                            dataKey="predicted"
                            stroke="#10b981"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            name={t('predictedYield')}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-400 flex items-center justify-center text-muted-foreground">
                        <div className="text-center">
                          <Milk className="h-16 w-16 mx-auto mb-4 opacity-50" />
                          <p>No milk yield data available yet.</p>
                          <p className="text-sm">Add cattle data and run predictions to see trends.</p>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

            </div>

            {/* Farm Reports Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="h-5 w-5 text-primary" />
                  <span>{t.farmReports}</span>
                </CardTitle>
                <CardDescription>Generate and export comprehensive farm reports</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex space-x-4">
                  <Button 
                    onClick={handleGenerateDashboardReport}
                    className="flex items-center space-x-2"
                    disabled={cattleData.length === 0}
                  >
                    <FileText className="h-4 w-4" />
                    <span>{t('generateReport')}</span>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="overview" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold tracking-tight">{t('cattleOverview')}</h2>
                <p className="text-muted-foreground">{t('overviewDescription')}</p>
              </div>
              <div className="flex space-x-2">
                <Button 
                  variant="outline"
                  className="flex items-center space-x-2"
                  onClick={handleReloadData}
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>{t('reloadData')}</span>
                </Button>
                <Button 
                  className="flex items-center space-x-2"
                  onClick={() => setActiveTab("input")}
                >
                  <Plus className="h-4 w-4" />
                  <span>{t('addNewCattle')}</span>
                </Button>
              </div>
            </div>

            {/* Welcome Message for Returning Users */}
            {user && cattleData.length > 0 && (
              <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <span className="font-semibold text-green-800">{t('allCattleDataLoaded')}</span>
                </div>
                <p className="text-green-700 text-sm mt-1">
                  {t('loadedRecords', { count: cattleData.length })} {cattleData.length} {t('cattleDataCount')}
                </p>
              </div>
            )}

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {cattleData.length > 0 ? (
                cattleData.map((cattle) => (
                  <CattleCard 
                    key={cattle.id} 
                    cattle={cattle} 
                    language={language} 
                    userInputData={formData}
                  />
                ))
              ) : (
                <div className="col-span-full text-center py-12">
                  <Cow className="h-16 w-16 mx-auto mb-4 opacity-50 text-muted-foreground" />
                  <h3 className="text-lg font-semibold text-muted-foreground mb-2">No Cattle Data Yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Start by adding cattle data in the Input tab to see your herd overview here.
                  </p>
                  <Button 
                    onClick={() => setActiveTab("input")}
                    className="bg-primary hover:bg-primary/90"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Add Your First Cattle
                  </Button>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Input Tab */}
          <TabsContent value="input">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Cow className="h-5 w-5 text-primary" />
                  <span>{t('addCattle')}</span>
                </CardTitle>
                <CardDescription>{t('inputDescription')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-6 md:grid-cols-2">
                  {/* Enhanced Basic Cow Information */}
                  <div className="p-6 bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-bold text-lg text-blue-800 flex items-center">
                        <div className="p-2 bg-blue-200 rounded-full mr-3">
                          <Cow className="h-6 w-6 text-blue-700" />
                        </div>
                        {t('basicCowInfo')}
                      </h3>
                      <div className="px-3 py-1 bg-blue-200 text-blue-800 text-xs font-medium rounded-full">
                        {t('essential')}
                      </div>
                    </div>
                    <p className="text-sm text-blue-700 mb-6 leading-relaxed">
                      {t('basicInfoDescription')}
                    </p>

                    <div className="grid gap-4 md:grid-cols-1">
                      <div className="space-y-3">
                        <Label htmlFor="breed" className="text-sm font-semibold text-blue-800">{t('breed')} *</Label>
                        <Select value={formData.breed} onValueChange={(value) => handleInputChange("breed", value)}>
                          <SelectTrigger className="border-2 border-blue-300 hover:border-blue-400 transition-colors h-12">
                            <SelectValue placeholder={t('selectBreed')} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Holstein" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🐄</span>
                                <div>
                                  <div className="font-medium">{t('holstein')}</div>
                                  <div className="text-xs text-gray-500">High milk production</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="Jersey" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🐮</span>
                                <div>
                                  <div className="font-medium">{t('jersey')}</div>
                                  <div className="text-xs text-gray-500">Rich milk quality</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="Guernsey" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🐄</span>
                                <div>
                                  <div className="font-medium">{t('guernsey')}</div>
                                  <div className="text-xs text-gray-500">Golden milk</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="Ayrshire" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🐮</span>
                                <div>
                                  <div className="font-medium">{t('ayrshire')}</div>
                                  <div className="text-xs text-gray-500">Hardy breed</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="Brown Swiss" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🐄</span>
                                <div>
                                  <div className="font-medium">{t('brownSwiss')}</div>
                                  <div className="text-xs text-gray-500">Dual purpose</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="Simmental" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🐮</span>
                                <div>
                                  <div className="font-medium">{t('simmental')}</div>
                                  <div className="text-xs text-gray-500">Large frame</div>
                                </div>
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="age_months" className="text-sm font-semibold text-blue-800">{t('ageMonths')} *</Label>
                        <Input
                          id="age_months"
                          type="number"
                          placeholder={t('enterAge')}
                          value={formData.age_months}
                          onChange={(e) => handleInputChange("age_months", parseFloat(e.target.value))}
                          className="border-2 border-blue-300 hover:border-blue-400 transition-colors h-12"
                        />
                        <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                          {t('ageRange')}: 24-120 months
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="weight_kg" className="text-sm font-semibold text-blue-800">{t('weightKg')} *</Label>
                        <Input
                          id="weight_kg"
                          type="number"
                          placeholder={t('enterWeight')}
                          value={formData.weight_kg}
                          onChange={(e) => handleInputChange("weight_kg", parseFloat(e.target.value))}
                          className="border-2 border-blue-300 hover:border-blue-400 transition-colors h-12"
                        />
                        <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                          {t('weightRange')}: 300-1200 kg
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="feed_quantity_kg" className="text-sm font-semibold text-blue-800">{t('feedQuantity')} *</Label>
                        <Input
                          id="feed_quantity_kg"
                          type="number"
                          placeholder={t('enterFeedQuantity')}
                          value={formData.feed_quantity_kg}
                          onChange={(e) => handleInputChange("feed_quantity_kg", parseFloat(e.target.value))}
                          className="border-2 border-blue-300 hover:border-blue-400 transition-colors h-12"
                        />
                        <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                          {t('feedRange')}: 5-50 kg per day
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="temperature" className="text-sm font-semibold text-blue-800">{t('temperature')} *</Label>
                        <Input
                          id="temperature"
                          type="number"
                          placeholder={t('enterTemperature')}
                          value={formData.temperature}
                          onChange={(e) => handleInputChange("temperature", parseFloat(e.target.value))}
                          className="border-2 border-blue-300 hover:border-blue-400 transition-colors h-12"
                        />
                        <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                          {t('tempRange')}: -10°C to 45°C
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="humidity" className="text-sm font-semibold text-blue-800">{t('humidity')} *</Label>
                        <Input
                          id="humidity"
                          type="number"
                          placeholder={t('enterHumidity')}
                          value={formData.humidity}
                          onChange={(e) => handleInputChange("humidity", parseFloat(e.target.value))}
                          className="border-2 border-blue-300 hover:border-blue-400 transition-colors h-12"
                        />
                        <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                          {t('humidityRange')}: 0-100%
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Enhanced Farm & Environment Details */}
                  <div className="p-6 bg-gradient-to-r from-purple-50 to-purple-100 border-2 border-purple-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-bold text-lg text-purple-800 flex items-center">
                        <div className="p-2 bg-purple-200 rounded-full mr-3">
                          <Globe className="h-6 w-6 text-purple-700" />
                        </div>
                        {t('farmEnvironmentDetails')}
                      </h3>
                      <div className="px-3 py-1 bg-purple-200 text-purple-800 text-xs font-medium rounded-full">
                        {t('environmental')}
                      </div>
                    </div>
                    <p className="text-sm text-purple-700 mb-6 leading-relaxed">
                      {t('environmentDescription')}
                    </p>

                    <div className="grid gap-4 md:grid-cols-1">
                      <div className="space-y-3">
                        <Label htmlFor="feed_type" className="text-sm font-semibold text-purple-800">{t('feedType')}</Label>
                        <Select value={formData.feed_type} onValueChange={(value) => handleInputChange("feed_type", value)}>
                          <SelectTrigger className="border-2 border-purple-300 hover:border-purple-400 transition-colors h-12">
                            <SelectValue placeholder={t('selectFeedType')} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="green_fodder" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🌱</span>
                                <div>
                                  <div className="font-medium">{t('greenFodder')}</div>
                                  <div className="text-xs text-gray-500">Fresh grass and leaves</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="dry_fodder" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🌾</span>
                                <div>
                                  <div className="font-medium">{t('dryFodder')}</div>
                                  <div className="text-xs text-gray-500">Hay and straw</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="concentrates" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🌽</span>
                                <div>
                                  <div className="font-medium">{t('concentrates')}</div>
                                  <div className="text-xs text-gray-500">Grains and pellets</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="silage" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🥬</span>
                                <div>
                                  <div className="font-medium">{t('silage')}</div>
                                  <div className="text-xs text-gray-500">Fermented feed</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="mixed" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🥗</span>
                                <div>
                                  <div className="font-medium">{t('mixed')}</div>
                                  <div className="text-xs text-gray-500">Combination feed</div>
                                </div>
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="grazing_hours" className="text-sm font-semibold text-purple-800">{t('grazingHours')}: {formData.grazing_hours} {t('hours')}</Label>
                        <Input
                          id="grazing_hours"
                          type="number"
                          step="0.1"
                          min="0"
                          max="12"
                          placeholder={t('dailyGrazingHours')}
                          value={formData.grazing_hours}
                          onChange={(e) => handleInputChange("grazing_hours", parseFloat(e.target.value))}
                          className="border-2 border-purple-300 hover:border-purple-400 transition-colors h-12"
                        />
                        <div className="text-xs text-purple-600 bg-purple-100 p-2 rounded">
                          {t('grazingRange')}: 0-12 hours per day
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="body_temperature" className="text-sm font-semibold text-purple-800">{t('bodyTemperature')}</Label>
                        <Input
                          id="body_temperature"
                          type="number"
                          step="0.1"
                          placeholder={t('bodyTemp')}
                          value={formData.body_temperature}
                          onChange={(e) => handleInputChange("body_temperature", parseFloat(e.target.value))}
                          className="border-2 border-purple-300 hover:border-purple-400 transition-colors h-12"
                        />
                        <div className="text-xs text-purple-600 bg-purple-100 p-2 rounded">
                          {t('bodyTempRange')}: 36-45°C (Normal: 38-39°C)
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label htmlFor="heart_rate" className="text-sm font-semibold text-purple-800">{t('heartRate')}</Label>
                        <Input
                          id="heart_rate"
                          type="number"
                          placeholder={t('enterHeartRate')}
                          value={formData.heart_rate}
                          onChange={(e) => handleInputChange("heart_rate", parseFloat(e.target.value))}
                          className="border-2 border-purple-300 hover:border-purple-400 transition-colors h-12"
                        />
                        <div className="text-xs text-purple-600 bg-purple-100 p-2 rounded">
                          {t('heartRateRange')}: 48-84 bpm (Normal: 60-70 bpm)
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Enhanced Health Check Sections */}
                <div className="space-y-6">
                  {/* Walking & Movement Check */}
                  <div className="p-6 bg-gradient-to-r from-orange-50 to-orange-100 border-2 border-orange-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-bold text-lg text-orange-800 flex items-center">
                        <div className="p-2 bg-orange-200 rounded-full mr-3">
                          <AlertTriangle className="h-6 w-6 text-orange-700" />
                        </div>
{t('walkingMovementAssessment')}
                      </h3>
                      <div className="px-3 py-1 bg-orange-200 text-orange-800 text-xs font-medium rounded-full">
{t('mobilityCheck')}
                      </div>
                    </div>
                    <p className="text-sm text-orange-700 mb-6 leading-relaxed">
{t('observeMovement')}
                    </p>
                    <div className="grid gap-6 md:grid-cols-2">
                      <div className="space-y-3">
                        <Label htmlFor="lameness_score" className="text-sm font-semibold text-orange-800">
{t('walkingConditionAssessment')}
                        </Label>
                        <Select 
                          value={formData.lameness_score?.toString()} 
                          onValueChange={(value) => handleInputChange("lameness_score", parseInt(value))}
                        >
                          <SelectTrigger className="border-2 border-orange-300 hover:border-orange-400 transition-colors h-12">
                            <SelectValue placeholder={t('selectWalkingCondition')} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😊</span>
                                <div>
                                  <div className="font-medium">{t('normalWalking')}</div>
                                  <div className="text-xs text-gray-500">{t('walksNormally')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="2" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🤔</span>
                                <div>
                                  <div className="font-medium">{t('slightDifference')}</div>
                                  <div className="text-xs text-gray-500">{t('walksDifferently')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="3" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😟</span>
                                <div>
                                  <div className="font-medium">{t('visibleLimping')}</div>
                                  <div className="text-xs text-gray-500">{t('clearlyLimping')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="4" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😰</span>
                                <div>
                                  <div className="font-medium">{t('reluctantMovement')}</div>
                                  <div className="text-xs text-gray-500">{t('doesntWantWalk')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="5" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😢</span>
                                <div>
                                  <div className="font-medium">{t('severeLameness')}</div>
                                  <div className="text-xs text-gray-500">{t('difficultyWalking')}</div>
                                </div>
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-3">
                        <Label htmlFor="walking_distance_km" className="text-sm font-semibold text-orange-800">
{t('dailyWalkingDistance')}
                        </Label>
                        <Input
                          id="walking_distance_km"
                          type="number"
                          step="0.1"
                          placeholder={t('walkingPlaceholder')}
                          value={formData.walking_distance_km}
                          onChange={(e) => handleInputChange("walking_distance_km", parseFloat(e.target.value))}
                          className="border-2 border-orange-300 hover:border-orange-400 transition-colors h-12"
                        />
                        <div className="text-xs text-orange-600 bg-orange-100 p-2 rounded">
{t('typicalRange')}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Eating & Digestion Check */}
                  <div className="p-6 bg-gradient-to-r from-yellow-50 to-yellow-100 border-2 border-yellow-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-bold text-lg text-yellow-800 flex items-center">
                        <div className="p-2 bg-yellow-200 rounded-full mr-3">
                          <Activity className="h-6 w-6 text-yellow-700" />
                        </div>
{t('eatingDigestionAssessment')}
                      </h3>
                      <div className="px-3 py-1 bg-yellow-200 text-yellow-800 text-xs font-medium rounded-full">
{t('nutritionCheck')}
                      </div>
                    </div>
                    <p className="text-sm text-yellow-700 mb-6 leading-relaxed">
{t('monitorAppetite')}
                    </p>
                    <div className="grid gap-6 md:grid-cols-2">
                      <div className="space-y-3">
                        <Label htmlFor="appetite_score" className="text-sm font-semibold text-yellow-800">
{t('appetiteEatingBehavior')}
                        </Label>
                        <Select 
                          value={formData.appetite_score?.toString()} 
                          onValueChange={(value) => handleInputChange("appetite_score", parseInt(value))}
                        >
                          <SelectTrigger className="border-2 border-yellow-300 hover:border-yellow-400 transition-colors h-12">
                            <SelectValue placeholder={t('selectEatingCondition')} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😢</span>
                                <div>
                                  <div className="font-medium">{t('poorAppetite')}</div>
                                  <div className="text-xs text-gray-500">{t('notEatingMuch')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="2" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😟</span>
                                <div>
                                  <div className="font-medium">{t('reducedAppetite')}</div>
                                  <div className="text-xs text-gray-500">{t('eatingLess')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="3" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">🤔</span>
                                <div>
                                  <div className="font-medium">{t('averageAppetite')}</div>
                                  <div className="text-xs text-gray-500">{t('eatingOkay')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="4" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😊</span>
                                <div>
                                  <div className="font-medium">{t('goodAppetite')}</div>
                                  <div className="text-xs text-gray-500">{t('eatingWell')}</div>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="5" className="py-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">😋</span>
                                <div>
                                  <div className="font-medium">{t('excellentAppetite')}</div>
                                  <div className="text-xs text-gray-500">{t('eatingVeryWell')}</div>
                                </div>
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-3">
                        <Label htmlFor="rumination_hours" className="text-sm font-semibold text-yellow-800">
{t('dailyRuminationHours')}
                        </Label>
                        <Input
                          id="rumination_hours"
                          type="number"
                          step="0.5"
                          min="0"
                          max="12"
                          placeholder={t('ruminationPlaceholder')}
                          value={formData.rumination_hours}
                          onChange={(e) => handleInputChange("rumination_hours", parseFloat(e.target.value))}
                          className="border-2 border-yellow-300 hover:border-yellow-400 transition-colors h-12"
                        />
                        <div className="text-xs text-yellow-600 bg-yellow-100 p-2 rounded">
{t('normalRuminationRange')}
                        </div>
                      </div>
                    </div>
                    <div className="mt-6 p-4 bg-gradient-to-r from-yellow-100 to-yellow-50 rounded-lg border border-yellow-200">
                      <p className="text-sm text-yellow-800 font-medium mb-2">{t('healthIndicators')}</p>
                      <ul className="text-xs text-yellow-700 space-y-1">
                        <li>{t('healthyCowsSpend')}</li>
                        <li>{t('lessThanFiveHours')}</li>
                        <li>{t('watchForSigns')}</li>
                      </ul>
                    </div>
                  </div>

                  {/* Overall Health & Appearance Check */}
                  <div className="p-4 bg-purple-50 border-2 border-purple-200 rounded-lg">
                    <h3 className="font-semibold text-purple-800 mb-2 flex items-center">
                      <CheckCircle className="h-5 w-5 mr-2" />
{t('overallHealthAppearance')}
                    </h3>
                    <p className="text-sm text-purple-700 mb-4">{t('checkGeneralHealth')}</p>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label htmlFor="coat_condition">{t('coatConditionLabel')}</Label>
                        <Select 
                          value={formData.coat_condition?.toString()} 
                          onValueChange={(value) => handleInputChange("coat_condition", parseInt(value))}
                        >
                          <SelectTrigger className="border-2 border-purple-300">
                            <SelectValue placeholder={t('chooseCoatCondition')} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1">{t('veryDullRough')}</SelectItem>
                            <SelectItem value="2">{t('somewhatDull')}</SelectItem>
                            <SelectItem value="3">{t('averageLooking')}</SelectItem>
                            <SelectItem value="4">{t('goodShineHealthy')}</SelectItem>
                            <SelectItem value="5">{t('veryShinyGlossy')}</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="body_temperature">{t('bodyTemperatureLabel')}</Label>
                        <Input
                          id="body_temperature"
                          type="number"
                          step="0.1"
                          placeholder={t('normalTempPlaceholder')}
                          value={formData.body_temperature}
                          onChange={(e) => handleInputChange("body_temperature", parseFloat(e.target.value))}
                          className="border-2 border-purple-300"
                        />
                      </div>
                    </div>
                    <div className="mt-4 p-3 bg-purple-100 rounded-lg">
                      <p className="text-xs text-purple-800">
{t('signsOfGoodHealth')}
                      </p>
                    </div>
                  </div>

                  {/* Udder & Milk Quality Check */}
                  <div className="p-4 bg-blue-50 border-2 border-blue-200 rounded-lg">
                    <h3 className="font-semibold text-blue-800 mb-2 flex items-center">
                      <Heart className="h-5 w-5 mr-2" />
{t('udderMilkQualityCheck')}
                    </h3>
                    <p className="text-sm text-blue-700 mb-4">{t('checkUdderHealth')}</p>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label htmlFor="udder_swelling">{t('udderSwellingLabel')}</Label>
                        <Select 
                          value={formData.udder_swelling?.toString()} 
                          onValueChange={(value) => handleInputChange("udder_swelling", parseInt(value))}
                        >
                          <SelectTrigger className="border-2 border-blue-300">
                            <SelectValue placeholder={t('chooseUdderCondition')} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="0">{t('normalSoftUdder')}</SelectItem>
                            <SelectItem value="1">{t('slightlySwollenUdder')}</SelectItem>
                            <SelectItem value="2">{t('clearlySwollenUdder')}</SelectItem>
                            <SelectItem value="3">{t('verySwollenUdder')}</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="heart_rate">{t('heartRateLabel')}</Label>
                        <Input
                          id="heart_rate"
                          type="number"
                          placeholder={t('normalHeartRatePlaceholder')}
                          value={formData.heart_rate}
                          onChange={(e) => handleInputChange("heart_rate", parseFloat(e.target.value))}
                          className="border-2 border-blue-300"
                        />
                      </div>
                    </div>
                    <div className="mt-4 p-3 bg-blue-100 rounded-lg">
                      <p className="text-xs text-blue-800">
{t('warningSignsUdder')}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Prediction Type Selection */}
                <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h3 className="font-semibold text-blue-800 mb-4">{t('aiPredictionServices')}</h3>
                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <Button 
                      onClick={handleMilkPrediction}
                      disabled={loadingMilk}
                      className="bg-green-600 hover:bg-green-700"
                    >
                      {loadingMilk ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
{t('predicting')}
                        </>
                      ) : (
                        <>
                          <Milk className="h-4 w-4 mr-2" />
{t('predictMilkYield')}
                        </>
                      )}
                    </Button>
                    
                    <Button 
                      onClick={handleDiseasePrediction}
                      disabled={loadingDisease}
                      className="bg-red-600 hover:bg-red-700"
                    >
                      {loadingDisease ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
{t('analyzing')}
                        </>
                      ) : (
                        <>
                          <Heart className="h-4 w-4 mr-2" />
{t('detectDisease')}
                        </>
                      )}
                    </Button>
                    
                    <Button 
                      onClick={handleBothPredictions}
                      disabled={loadingMilk || loadingDisease}
                      className="bg-purple-600 hover:bg-purple-700"
                    >
                      {(loadingMilk || loadingDisease) ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
{t('processing')}
                        </>
                      ) : (
                        <>
                          <Activity className="h-4 w-4 mr-2" />
{t('completeAnalysis')}
                        </>
                      )}
                    </Button>

                    <Button 
                      onClick={handleSaveCattleData}
                      disabled={isSavingData}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      {isSavingData ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Saving...
                        </>
                      ) : (
                        <>
                          <Plus className="h-4 w-4 mr-2" />
{t("saveCattleData")}
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {/* Milk Yield Prediction Results */}
                {milkPrediction && (
                  <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h3 className="font-semibold text-green-800 mb-2 flex items-center">
                      <Milk className="h-5 w-5 mr-2" />
                      {t("milkYieldPredictionResults")}
                    </h3>
                    <div className="grid gap-2">
                      <p className="text-lg font-bold text-green-700">
                        {t("predictedMilkYield")}: {milkPrediction.predicted_milk_yield?.toFixed(2)} L/{t("day")}
                      </p>
                      <p className="text-sm text-green-600">{t("status")}: {milkPrediction.status}</p>
                      <p className="text-xs text-green-500">{t("timestamp")}: {milkPrediction.timestamp}</p>
                      {milkPrediction.validation_warnings && Array.isArray(milkPrediction.validation_warnings) && milkPrediction.validation_warnings.length > 0 && (
                        <div className="mt-2">
                          <p className="text-sm font-medium text-yellow-700">{t("warnings")}:</p>
                          <ul className="text-xs text-yellow-600 list-disc list-inside">
                            {milkPrediction.validation_warnings.map((warning, index) => (
                              <li key={index}>{typeof warning === 'string' ? warning : JSON.stringify(warning)}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Disease Detection Results */}
                {diseasePrediction && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <h3 className="font-semibold text-red-800 mb-2 flex items-center">
                      <Heart className="h-5 w-5 mr-2" />
                      {t("diseaseDetectionResults")}
                    </h3>
                    <div className="grid gap-3">
                      <div className="flex items-center justify-between">
                        <p className="text-lg font-bold text-red-700">
                          {t("healthStatus")}: {diseasePrediction.predicted_disease}
                        </p>
                        <Badge className={
                          diseasePrediction.predicted_disease === 'healthy' 
                            ? 'bg-green-100 text-green-800' 
                            : diseasePrediction.risk_level === 'high'
                            ? 'bg-red-100 text-red-800'
                            : diseasePrediction.risk_level === 'medium'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-blue-100 text-blue-800'
                        }>
                          {diseasePrediction.risk_level} {t("risk")}
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-red-600">
                        {t("confidence")}: {(diseasePrediction.confidence * 100).toFixed(1)}%
                      </p>
                      
                      {diseasePrediction.recommendations && Array.isArray(diseasePrediction.recommendations) && diseasePrediction.recommendations.length > 0 && (
                        <div className="mt-3">
                          <p className="text-sm font-medium text-red-700 mb-2">{t("recommendations")}:</p>
                          <ul className="text-xs text-red-600 list-disc list-inside space-y-1">
                            {diseasePrediction.recommendations.map((rec, index) => {
                              // Translate common recommendation text
                              let translatedRec = rec;
                              if (typeof rec === 'string') {
                                if (rec.includes('Continue current management practices')) {
                                  translatedRec = t("continueCurrentPractices");
                                } else if (rec.includes('Monitor regularly for any changes in health parameters')) {
                                  translatedRec = t("monitorRegularly");
                                } else if (rec.includes('Maintain good nutrition and hygiene')) {
                                  translatedRec = t("maintainNutrition");
                                } else {
                                  translatedRec = rec; // Keep original if no translation found
                                }
                              }
                              return <li key={index}>{translatedRec}</li>
                            })}
                          </ul>
                        </div>
                      )}
                      
                      <p className="text-xs text-red-500">{t("timestamp")}: {diseasePrediction.timestamp}</p>
                    </div>
                  </div>
                )}

                {/* Error Display */}
                {error && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <h3 className="font-semibold text-red-800 mb-2">Error</h3>
                    <p className="text-red-700">{typeof error === 'string' ? error : JSON.stringify(error)}</p>
                  </div>
                )}

                {/* Save Data Dialog */}
                {(milkPrediction || diseasePrediction) && !savedCattleId && (
                  <div className="mt-6 p-6 bg-blue-50 border-2 border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="font-semibold text-blue-800 mb-2">{t("saveCattleDataToDatabase")}</h3>
                        <p className="text-blue-700 text-sm">
                          {t("predictionsReady")}
                        </p>
                      </div>
                      <Button
                        onClick={handleSaveCattleData}
                        disabled={isSavingData}
                        className="bg-blue-600 hover:bg-blue-700"
                      >
                        {isSavingData ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                            Saving...
                          </>
                        ) : (
                          <>
                            <CheckCircle className="h-4 w-4 mr-2" />
                            Save to Database
                          </>
                        )}
                      </Button>
                    </div>
                    <div className="text-xs text-blue-600 bg-blue-100 p-3 rounded">
                      💡 This will save all cattle information and link it with your predictions for comprehensive reporting.
                    </div>
                  </div>
                )}

                <div className="pt-6 border-t">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground mb-4">
                      Use the AI Prediction Services panel above to get milk yield predictions and disease detection results.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Validation Error Dialog */}
      <Dialog open={showValidationDialog} onOpenChange={setShowValidationDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invalid Input</DialogTitle>
            <DialogDescription>
              {validationError}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              onClick={() => setShowValidationDialog(false)}
              className="w-full"
            >
              OK
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
    </ProtectedRoute>
  )
}
