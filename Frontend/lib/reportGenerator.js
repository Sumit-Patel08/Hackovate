import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'
import { supabase } from './supabase'

export class ReportGenerator {
  constructor(language = 'en') {
    try {
      this.doc = null
      this.language = language
      // Use hardcoded English text to avoid translation errors
      this.t = {
        farmSummary: 'Farm Summary',
        productionAnalysis: 'Production Analysis', 
        healthOverview: 'Health Overview',
        recommendations: 'Recommendations',
        cattleHealthMilkReport: 'Cattle Health & Milk Report',
        comprehensiveCattleReport: 'Comprehensive Cattle Report',
        dairyCattleMonitoring: 'Dairy Cattle Monitoring',
        aiPoweredPlatform: 'AI-Powered Platform',
        cattleInfo: 'Cattle Information',
        milkYieldPrediction: 'Milk Yield Prediction',
        diseaseDetection: 'Disease Detection',
        inputData: 'Input Data',
        predictionResults: 'Prediction Results',
        healthMetrics: 'Health Metrics',
        environmentalFactors: 'Environmental Factors',
        notSpecified: 'Not Specified',
        months: 'months',
        feedType: 'Feed Type',
        feedQuantity: 'Feed Quantity',
        kgPerDay: 'kg/day',
        grazingHours: 'Grazing Hours',
        hoursPerDay: 'hours/day',
        bodyTemperature: 'Body Temperature',
        heartRate: 'Heart Rate',
        bpm: 'BPM',
        environmentalTemperature: 'Environmental Temperature',
        humidity: 'Humidity',
        cattleId: 'Cattle ID',
        generatedOn: 'Generated On',
        breed: 'Breed',
        age: 'Age',
        weight: 'Weight',
        dailyYield: 'Daily Yield',
        healthScore: 'Health Score',
        status: 'Status',
        good: 'Good',
        fair: 'Fair', 
        poor: 'Poor',
        individualCattleSummary: 'Individual Cattle Summary',
        farmReports: 'Farm Reports',
        totalCattle: 'Total Cattle',
        dailyMilkProduction: 'Daily Milk Production',
        averageYieldPerCattle: 'Average Yield per Cattle',
        projectedMonthlyProduction: 'Projected Monthly',
        mostCommonBreed: 'Most Common Breed',
        lowProducers: 'Low Producers (<15L)',
        mediumProducers: 'Medium Producers (15-25L)',
        highProducers: 'High Producers (>25L)',
        productionEfficiency: 'Production Efficiency',
        excellentHealth: 'Excellent Health (90-100)',
        goodHealth: 'Good Health (80-89)',
        fairHealth: 'Fair Health (60-79)',
        poorHealth: 'Poor Health (<60)',
        overallHealthRate: 'Overall Health Score',
        healthyCattle: 'Disease-Free Cattle',
        farmRecommendations: 'Farm Recommendations'
      }
      console.log('ReportGenerator constructor completed successfully')
    } catch (error) {
      console.error('Error in ReportGenerator constructor:', error)
      throw error
    }
  }

  // Generate comprehensive PDF report
  async generatePDFReport(cattleData, predictions, userInput) {
    this.doc = new jsPDF()
    
    // Set up document properties
    this.doc.setProperties({
      title: `Cattle Health & Milk Report - ${cattleData.cattle_id}`,
      subject: 'Comprehensive Cattle Report',
      author: 'Dairy Cattle Monitoring',
      creator: 'AI-Powered Platform'
    })

    // Add header
    this.addHeader(cattleData)
    
    // Add cattle information
    this.addCattleInfo(cattleData, userInput)
    
    // Add AI predictions section
    this.addPredictionsSection(predictions)
    
    // Add recommendations
    this.addRecommendations(predictions)
    
    // Add footer
    this.addFooter()
    
    return this.doc
  }

  addHeader(cattleData) {
    const pageWidth = this.doc.internal.pageSize.width
    
    // Title
    this.doc.setFontSize(20)
    this.doc.setFont(undefined, 'bold')
    this.doc.text('CATTLE HEALTH & MILK REPORT', pageWidth / 2, 20, { align: 'center' })
    
    // Subtitle
    this.doc.setFontSize(14)
    this.doc.setFont(undefined, 'normal')
    this.doc.text(`Cattle ID: ${cattleData.cattle_id}`, pageWidth / 2, 30, { align: 'center' })
    
    // Date
    this.doc.setFontSize(10)
    this.doc.text(`Generated On: ${new Date().toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })}`, pageWidth / 2, 40, { align: 'center' })
    
    // Line separator
    this.doc.setLineWidth(0.5)
    this.doc.line(20, 45, pageWidth - 20, 45)
  }

  addCattleInfo(cattleData, userInput) {
    let yPos = 55
    
    // Section title
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.cattleInfo.toUpperCase(), 20, yPos)
    yPos += 15
    
    // Basic info - Use actual cattle data from Supabase
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const info = [
      [`${this.t.breed}: ${cattleData?.breed || userInput?.breed || this.t.notSpecified}`, `${this.t.age}: ${cattleData?.age_months || userInput?.age_months || this.t.notSpecified} ${this.t.months}`],
      [`${this.t.weight}: ${cattleData?.weight_kg || userInput?.weight_kg || this.t.notSpecified} kg`, `${this.t.feedType}: ${cattleData?.feed_type || userInput?.feed_type || this.t.notSpecified}`],
      [`${this.t.feedQuantity}: ${cattleData?.feed_quantity_kg || userInput?.feed_quantity_kg || this.t.notSpecified} ${this.t.kgPerDay}`, `${this.t.grazingHours}: ${cattleData?.grazing_hours || userInput?.grazing_hours || this.t.notSpecified} ${this.t.hoursPerDay}`],
      [`${this.t.bodyTemperature}: ${cattleData?.body_temperature || userInput?.body_temperature || this.t.notSpecified}°C`, `${this.t.heartRate}: ${cattleData?.heart_rate || userInput?.heart_rate || this.t.notSpecified} ${this.t.bpm}`],
      [`${this.t.environmentalTemperature}: ${cattleData?.environmental_data?.temperature || userInput?.temperature || this.t.notSpecified}°C`, `${this.t.humidity}: ${cattleData?.environmental_data?.humidity || userInput?.humidity || this.t.notSpecified}%`]
    ]
    
    info.forEach(([left, right]) => {
      this.doc.setFont(undefined, 'normal')
      this.doc.text(left, 20, yPos)
      this.doc.text(right, 110, yPos)
      yPos += 8
    })
  }

  addPredictionsSection(predictions) {
    let yPos = 180
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.predictionResults.toUpperCase(), 20, yPos)
    yPos += 15
    
    // Get milk and disease predictions from the data structure
    const milkPredictions = predictions?.allPredictions?.filter(p => p.prediction_type === 'milk_yield') || []
    const diseasePredictions = predictions?.allPredictions?.filter(p => p.prediction_type === 'disease') || []
    
    // Milk Yield Prediction Section
    if (milkPredictions.length > 0 || predictions?.milkYield) {
      const milkPrediction = milkPredictions[0] || null
      const milkData = predictions?.milkYield || milkPrediction?.prediction_result
      
      this.doc.setFontSize(14)
      this.doc.setFont(undefined, 'bold')
      this.doc.text(this.t.milkYieldPrediction.toUpperCase(), 20, yPos)
      yPos += 10
      
      this.doc.setFontSize(11)
      this.doc.setFont(undefined, 'normal')
      
      if (milkData) {
        // Prediction Results
        this.doc.text(`Predicted Daily Yield: ${milkData.predicted_milk_yield || milkData.prediction || 'N/A'} liters`, 25, yPos)
        yPos += 8
        this.doc.text(`Confidence Level: ${milkData.confidence ? (milkData.confidence * 100).toFixed(1) : 'N/A'}%`, 25, yPos)
        yPos += 8
        
        if (milkPrediction?.created_at) {
          this.doc.text(`Prediction Date: ${new Date(milkPrediction.created_at).toLocaleString()}`, 25, yPos)
          yPos += 8
        }
        
        // Input Data Used for Prediction
        if (milkPrediction?.input_data) {
          this.doc.setFont(undefined, 'bold')
          this.doc.text('Input Data Used for Prediction:', 25, yPos)
          yPos += 6
          this.doc.setFont(undefined, 'normal')
          
          const input = milkPrediction.input_data
          if (input.breed) this.doc.text(`• Breed: ${input.breed}`, 30, yPos), yPos += 6
          if (input.age_months) this.doc.text(`• Age: ${input.age_months} months (${Math.floor(input.age_months / 12)} years)`, 30, yPos), yPos += 6
          if (input.weight_kg) this.doc.text(`• Weight: ${input.weight_kg} kg`, 30, yPos), yPos += 6
          if (input.feed_type) this.doc.text(`• Feed Type: ${input.feed_type}`, 30, yPos), yPos += 6
          if (input.feed_quantity_kg) this.doc.text(`• Feed Quantity: ${input.feed_quantity_kg} kg/day`, 30, yPos), yPos += 6
          if (input.grazing_hours) this.doc.text(`• Grazing Hours: ${input.grazing_hours} hours/day`, 30, yPos), yPos += 6
          if (input.body_temperature) this.doc.text(`• Body Temperature: ${input.body_temperature}°C`, 30, yPos), yPos += 6
          if (input.heart_rate) this.doc.text(`• Heart Rate: ${input.heart_rate} BPM`, 30, yPos), yPos += 6
          if (input.temperature) this.doc.text(`• Environmental Temperature: ${input.temperature}°C`, 30, yPos), yPos += 6
          if (input.humidity) this.doc.text(`• Humidity: ${input.humidity}%`, 30, yPos), yPos += 6
        }
      } else {
        this.doc.text('No milk yield prediction data available', 25, yPos)
        yPos += 8
      }
      yPos += 10
    }
    
    // Disease Detection Section
    if (diseasePredictions.length > 0 || predictions?.disease) {
      const diseasePrediction = diseasePredictions[0] || null
      const diseaseData = predictions?.disease || diseasePrediction?.prediction_result
      
      this.doc.setFontSize(14)
      this.doc.setFont(undefined, 'bold')
      this.doc.text(this.t.diseaseDetection.toUpperCase(), 20, yPos)
      yPos += 10
      
      this.doc.setFontSize(11)
      this.doc.setFont(undefined, 'normal')
      
      if (diseaseData) {
        // Disease Prediction Results
        this.doc.text(`Predicted Disease: ${diseaseData.predicted_disease || 'N/A'}`, 25, yPos)
        yPos += 8
        this.doc.text(`Risk Level: ${diseaseData.risk_level || 'N/A'}`, 25, yPos)
        yPos += 8
        this.doc.text(`Confidence: ${diseaseData.confidence ? (diseaseData.confidence * 100).toFixed(1) : 'N/A'}%`, 25, yPos)
        yPos += 8
        
        if (diseasePrediction?.created_at) {
          this.doc.text(`Prediction Date: ${new Date(diseasePrediction.created_at).toLocaleString()}`, 25, yPos)
          yPos += 8
        }
        
        // Health Metrics Input Data
        if (diseasePrediction?.input_data) {
          this.doc.setFont(undefined, 'bold')
          this.doc.text('Health Metrics Used for Analysis:', 25, yPos)
          yPos += 6
          this.doc.setFont(undefined, 'normal')
          
          const input = diseasePrediction.input_data
          if (input.white_blood_cells) this.doc.text(`• White Blood Cells: ${input.white_blood_cells} cells/μL`, 30, yPos), yPos += 6
          if (input.somatic_cell_count) this.doc.text(`• Somatic Cell Count: ${input.somatic_cell_count} cells/mL`, 30, yPos), yPos += 6
          if (input.rumen_ph) this.doc.text(`• Rumen pH: ${input.rumen_ph}`, 30, yPos), yPos += 6
          if (input.rumen_temperature) this.doc.text(`• Rumen Temperature: ${input.rumen_temperature}°C`, 30, yPos), yPos += 6
          if (input.lameness_score) this.doc.text(`• Lameness Score: ${input.lameness_score}/5`, 30, yPos), yPos += 6
          if (input.appetite_score) this.doc.text(`• Appetite Score: ${input.appetite_score}/5`, 30, yPos), yPos += 6
          if (input.coat_condition) this.doc.text(`• Coat Condition: ${input.coat_condition}/5`, 30, yPos), yPos += 6
          if (input.udder_swelling) this.doc.text(`• Udder Swelling Level: ${input.udder_swelling}/3`, 30, yPos), yPos += 6
        }
      } else {
        this.doc.text('No disease prediction data available', 25, yPos)
        yPos += 8
      }
      yPos += 10
    }
    
    // Health Score Analysis
    if (predictions?.cattle?.healthScore) {
      const healthScore = predictions.cattle.healthScore
      
      this.doc.setFontSize(14)
      this.doc.setFont(undefined, 'bold')
      this.doc.text(this.t.healthMetrics.toUpperCase(), 20, yPos)
      yPos += 10
      
      this.doc.setFontSize(11)
      this.doc.setFont(undefined, 'normal')
      this.doc.text(`Overall Health Score: ${healthScore}%`, 25, yPos)
      yPos += 8
      
      // Health status interpretation
      let healthStatus = 'Poor'
      if (healthScore >= 90) healthStatus = 'Excellent'
      else if (healthScore >= 80) healthStatus = 'Good'
      else if (healthScore >= 70) healthStatus = 'Fair'
      else if (healthScore >= 60) healthStatus = 'Below Average'
      
      this.doc.text(`Health Status: ${healthStatus}`, 25, yPos)
      yPos += 8
      
      // Add health recommendations based on score
      if (healthScore < 70) {
        this.doc.text('⚠️ Health concerns detected - veterinary consultation recommended', 25, yPos)
        yPos += 8
      }
      yPos += 10
    }
    
    // Summary of all predictions
    if (predictions?.allPredictions && predictions.allPredictions.length > 0) {
      this.doc.setFontSize(14)
      this.doc.setFont(undefined, 'bold')
      this.doc.text('PREDICTION SUMMARY', 20, yPos)
      yPos += 10
      
      this.doc.setFontSize(11)
      this.doc.setFont(undefined, 'normal')
      this.doc.text(`Total Predictions Generated: ${predictions.allPredictions.length}`, 25, yPos)
      yPos += 8
      
      predictions.allPredictions.forEach((pred, index) => {
        const date = new Date(pred.created_at).toLocaleDateString()
        this.doc.text(`${index + 1}. ${pred.prediction_type.toUpperCase()} - ${date}`, 25, yPos)
        yPos += 6
      })
    }
  }

  addRecommendations(predictions) {
    let yPos = 260
    
    // Check if we need a new page
    if (yPos > 250) {
      this.doc.addPage()
      yPos = 30
    }
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.recommendations.toUpperCase(), 20, yPos)
    yPos += 15
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const recommendations = []
    
    // AI-Driven Milk Yield Recommendations
    const milkData = predictions?.milkYield || predictions?.milkPrediction?.prediction_result
    if (milkData?.predicted_milk_yield) {
      const milkYield = parseFloat(milkData.predicted_milk_yield)
      const confidence = milkData.confidence || 0
      
      recommendations.push('=== AI-DRIVEN MILK PRODUCTION RECOMMENDATIONS ===')
      
      if (milkYield < 15) {
        recommendations.push('• URGENT: Low milk yield detected - immediate intervention required')
        recommendations.push('• Check for mastitis, nutritional deficiencies, or metabolic disorders')
        recommendations.push('• Increase energy-dense feed and ensure adequate protein intake')
        recommendations.push('• Monitor body condition score and adjust feeding accordingly')
      } else if (milkYield < 20) {
        recommendations.push('• Below average milk yield - optimization opportunities identified')
        recommendations.push('• Improve feed quality with higher energy and protein content')
        recommendations.push('• Review mineral supplementation (calcium, phosphorus, magnesium)')
        recommendations.push('• Assess environmental stress factors (heat, humidity, housing)')
      } else if (milkYield > 30) {
        recommendations.push('• Excellent milk production detected - maintain high standards')
        recommendations.push('• Monitor for metabolic stress and negative energy balance')
        recommendations.push('• Ensure adequate calcium supplementation to prevent milk fever')
        recommendations.push('• Implement precision feeding to sustain high production')
      } else {
        recommendations.push('• Good milk production levels - minor optimizations possible')
        recommendations.push('• Consider gradual feed quality improvements')
        recommendations.push('• Monitor for seasonal variations and adjust management')
      }
      
      if (confidence < 0.7) {
        recommendations.push('• Prediction confidence is moderate - collect more data for accuracy')
        recommendations.push('• Implement daily milk recording for better AI model training')
      }
    }
    
    // AI-Driven Disease Prevention & Health Recommendations
    const diseaseData = predictions?.disease || predictions?.diseasePrediction?.prediction_result
    const healthScore = predictions?.healthScore || predictions?.cattle?.healthScore
    
    recommendations.push('')
    recommendations.push('=== AI-DRIVEN HEALTH & DISEASE RECOMMENDATIONS ===')
    
    if (diseaseData?.predicted_disease && diseaseData.predicted_disease !== 'healthy') {
      const riskLevel = diseaseData.risk_level
      const confidence = diseaseData.confidence || 0
      
      if (riskLevel === 'high') {
        recommendations.push('• CRITICAL: High disease risk detected - immediate action required')
        recommendations.push('• Contact veterinarian immediately for professional diagnosis')
        recommendations.push('• Isolate animal to prevent disease spread')
        recommendations.push('• Implement strict biosecurity measures')
        recommendations.push('• Monitor vital signs every 4-6 hours')
      } else if (riskLevel === 'medium') {
        recommendations.push('• MODERATE: Disease risk identified - preventive measures needed')
        recommendations.push('• Schedule veterinary examination within 24-48 hours')
        recommendations.push('• Increase monitoring frequency for symptoms')
        recommendations.push('• Review vaccination schedule and update if necessary')
        recommendations.push('• Enhance hygiene and sanitation protocols')
      } else {
        recommendations.push('• LOW: Minor health concerns detected - preventive care advised')
        recommendations.push('• Continue regular health monitoring')
        recommendations.push('• Maintain current preventive health measures')
        recommendations.push('• Consider nutritional supplements for immune support')
      }
      
      if (confidence > 0.8) {
        recommendations.push('• High AI confidence in disease prediction - prioritize immediate action')
      }
    } else {
      recommendations.push('• Animal appears healthy according to AI analysis')
      recommendations.push('• Continue current health management practices')
      recommendations.push('• Maintain regular preventive care schedule')
    }
    
    // Health Score Based Recommendations
    if (healthScore) {
      if (healthScore < 60) {
        recommendations.push('• POOR health score detected - comprehensive health assessment needed')
        recommendations.push('• Review all management practices (nutrition, housing, hygiene)')
        recommendations.push('• Consider blood work and detailed veterinary examination')
      } else if (healthScore < 80) {
        recommendations.push('• FAIR health score - room for improvement identified')
        recommendations.push('• Focus on stress reduction and comfort improvements')
        recommendations.push('• Optimize nutrition and environmental conditions')
      } else if (healthScore >= 90) {
        recommendations.push('• EXCELLENT health score - maintain current management')
        recommendations.push('• Use as benchmark for other animals in the herd')
      }
    }
    
    // AI Model Insights and Future Predictions
    recommendations.push('')
    recommendations.push('=== AI MODEL INSIGHTS & FUTURE MONITORING ===')
    recommendations.push('• Continue data collection for improved AI predictions')
    recommendations.push('• Implement IoT sensors for real-time health monitoring')
    recommendations.push('• Schedule follow-up AI analysis in 7-14 days')
    recommendations.push('• Compare predictions with actual outcomes for model validation')
    
    // Custom AI recommendations if available
    if (diseaseData?.ai_recommendations && Array.isArray(diseaseData.ai_recommendations)) {
      recommendations.push('')
      recommendations.push('=== CUSTOM AI RECOMMENDATIONS ===')
      diseaseData.ai_recommendations.forEach(rec => {
        recommendations.push(`• ${rec}`)
      })
    }
    
    // General best practices
    if (recommendations.length < 10) {
      recommendations.push('')
      recommendations.push('=== GENERAL BEST PRACTICES ===')
      recommendations.push('• Maintain detailed daily records for AI model improvement')
      recommendations.push('• Ensure consistent feeding times and quality feed')
      recommendations.push('• Provide adequate fresh water access (80-120L/day)')
      recommendations.push('• Monitor environmental conditions (temperature, humidity)')
      recommendations.push('• Implement regular hoof trimming and health checks')
    }
    
    recommendations.forEach(rec => {
      if (rec.startsWith('===')) {
        this.doc.setFont(undefined, 'bold')
        this.doc.text(rec, 20, yPos)
        this.doc.setFont(undefined, 'normal')
        yPos += 8
      } else {
        const lines = this.doc.splitTextToSize(rec, 170)
        lines.forEach(line => {
          this.doc.text(line, 20, yPos)
          yPos += 6
        })
        yPos += 2
      }
    })
  }

  addFooter() {
    const pageHeight = this.doc.internal.pageSize.height
    const pageWidth = this.doc.internal.pageSize.width
    
    this.doc.setFontSize(8)
    this.doc.setFont(undefined, 'italic')
    this.doc.text(this.t.reportDisclaimer, 
                  pageWidth / 2, pageHeight - 20, { align: 'center' })
    
    this.doc.setFont(undefined, 'normal')
    this.doc.text(this.t.copyrightNotice, pageWidth / 2, pageHeight - 10, { align: 'center' })
  }

  // Save report to Supabase
  async generateFarmReport(userId, language = 'en') {
    console.log('ReportGenerator: Starting farm report generation for user:', userId)
    
    try {
      // Import CattleDataManager to get dynamic data
      const { CattleDataManager } = await import('./cattleDataManager')
      
      // Fetch all cattle data with predictions from Supabase
      const allCattleData = await CattleDataManager.getCattleWithPredictions(userId)
      console.log('ReportGenerator: Fetched dynamic cattle data:', allCattleData)
      
      if (!allCattleData || allCattleData.length === 0) {
        console.log('ReportGenerator: No cattle data found for user')
        throw new Error('No cattle data found for this user. Please add cattle data first.')
      }
      
      // Initialize PDF
      this.doc = new jsPDF()
      this.t = translations[language] || translations.en
      
      // Add report sections with dynamic data
      this.addReportHeader()
      this.addFarmSummary(allCattleData)
      this.addProductionAnalysis(allCattleData)
      this.addHealthOverview(allCattleData)
      this.addCattleSummaryTable(allCattleData)
      this.addFarmRecommendations(allCattleData)
      
      console.log('ReportGenerator: Farm report generated successfully with dynamic data')
      return this.doc
    } catch (error) {
      console.error('ReportGenerator: Error generating farm report:', error)
      throw error
    }
  }

  // Generate comprehensive dashboard report for all cattle
  async generateDashboardReport(allCattleData, userId) {
    try {
      console.log('ReportGenerator: Starting dashboard report generation')
      console.log('ReportGenerator: Cattle data received:', allCattleData)
      
      if (!allCattleData || allCattleData.length === 0) {
        throw new Error('No cattle data available for report generation')
      }
      
      console.log('Creating new jsPDF instance...')
      this.doc = new jsPDF()
      console.log('jsPDF instance created successfully')
      
      // Set up document properties
      this.doc.setProperties({
        title: `Farm Dashboard Report - ${new Date().toLocaleDateString()}`,
        subject: 'Complete Farm Analysis',
        author: 'Dairy Cattle Monitoring System',
        creator: 'AI-Powered Platform',
        reportDisclaimer: 'This report is generated by AI and should be used for guidance only. Consult a veterinarian for medical decisions.',
        copyrightNotice: '© 2024 Dairy Cattle Monitoring System - AI-Powered Platform'
      })
      console.log('Document properties set')

      // Add header
      console.log('Adding dashboard header...')
      this.addDashboardHeader(allCattleData.length)
      
      // Add farm summary
      console.log('Adding farm summary...')
      this.addFarmSummary(allCattleData)
      
      // Add production analysis
      console.log('Adding production analysis...')
      this.addProductionAnalysis(allCattleData)
      
      // Add health overview
      console.log('Adding health overview...')
      this.addHealthOverview(allCattleData)
      
      // Add individual cattle summary
      console.log('Adding cattle summary table...')
      this.addCattleSummaryTable(allCattleData)
      
      // Add recommendations
      console.log('Adding farm recommendations...')
      this.addFarmRecommendations(allCattleData)
      
      // Add footer
      console.log('Adding footer...')
      this.addFooter()
      
      console.log('ReportGenerator: Dashboard report generated successfully')
      return this.doc
    } catch (error) {
      console.error('Error in generateDashboardReport:', error)
      console.error('Error details:', error.message, error.stack)
      throw new Error(`Report generation failed: ${error.message}`)
    }
  }

  addDashboardHeader(totalCattle) {
    const pageWidth = this.doc.internal.pageSize.width
    
    // Title
    this.doc.setFontSize(20)
    this.doc.setFont(undefined, 'bold')
    this.doc.text('FARM DASHBOARD REPORT', pageWidth / 2, 20, { align: 'center' })
    
    // Subtitle
    this.doc.setFontSize(14)
    this.doc.setFont(undefined, 'normal')
    this.doc.text(`Complete Farm Analysis - ${totalCattle} Cattle`, pageWidth / 2, 30, { align: 'center' })
    
    // Date
    this.doc.setFontSize(10)
    this.doc.text(`Generated On: ${new Date().toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })}`, pageWidth / 2, 40, { align: 'center' })
    
    // Line separator
    this.doc.setLineWidth(0.5)
    this.doc.line(20, 45, pageWidth - 20, 45)
  }

  addFarmSummary(allCattleData) {
    let yPos = 55
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.farmSummary.toUpperCase(), 20, yPos)
    yPos += 15
    
    const totalCattle = allCattleData.length
    
    // Calculate milk production from actual AI predictions (dailyYield from cattleDataManager)
    const totalMilkProduction = allCattleData.reduce((sum, cattle) => 
      sum + (parseFloat(cattle.dailyYield) || 0), 0)
    const avgYieldPerCattle = totalCattle > 0 ? totalMilkProduction / totalCattle : 0
    const projectedMonthly = totalMilkProduction * 30
    
    // Get most common breed from actual cattle data
    const breedCounts = {}
    allCattleData.forEach(cattle => {
      const breed = cattle.breed || 'Unknown'
      breedCounts[breed] = (breedCounts[breed] || 0) + 1
    })
    const mostCommonBreed = Object.keys(breedCounts).length > 0 ? 
      Object.keys(breedCounts).reduce((a, b) => breedCounts[a] > breedCounts[b] ? a : b) : 'Unknown'
    
    // Calculate average age from cattle data
    const avgAge = allCattleData.length > 0 ? 
      allCattleData.reduce((sum, cattle) => sum + (cattle.age_months || 0), 0) / allCattleData.length : 0
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const summaryData = [
      [`${this.t.totalCattle}: ${totalCattle}`, `${this.t.dailyMilkProduction}: ${totalMilkProduction.toFixed(1)}L`],
      [`${this.t.averageYieldPerCattle}: ${avgYieldPerCattle.toFixed(1)}L`, `${this.t.projectedMonthlyProduction}: ${projectedMonthly.toFixed(0)}L`],
      [`${this.t.mostCommonBreed}: ${mostCommonBreed}`, `Average ${this.t.age}: ${Math.floor(avgAge / 12)} years`]
    ]
    
    summaryData.forEach(([label, value]) => {
      this.doc.setFont(undefined, 'bold')
      this.doc.text(label, 20, yPos)
      this.doc.setFont(undefined, 'normal')
      this.doc.text(value, 100, yPos)
      yPos += 8
    })
  }

  addProductionAnalysis(allCattleData) {
    let yPos = 120
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.productionAnalysis.toUpperCase(), 20, yPos)
    yPos += 15
    
    // Categorize cattle by production levels using actual AI predictions (dailyYield)
    const lowProducers = allCattleData.filter(cattle => 
      parseFloat(cattle.dailyYield) < 15).length
    const mediumProducers = allCattleData.filter(cattle => {
      const milkYield = parseFloat(cattle.dailyYield)
      return milkYield >= 15 && milkYield <= 25
    }).length
    const highProducers = allCattleData.filter(cattle => 
      parseFloat(cattle.dailyYield) > 25).length
    
    const efficiency = allCattleData.length > 0 ? 
      ((highProducers / allCattleData.length) * 100).toFixed(1) : 0
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const productionData = [
      [`${this.t.lowProducers}: ${lowProducers}`, `${this.t.mediumProducers}: ${mediumProducers}`],
      [`${this.t.highProducers}: ${highProducers}`, `${this.t.productionEfficiency}: ${efficiency}%`]
    ]
    
    productionData.forEach(([label, value]) => {
      this.doc.setFont(undefined, 'bold')
      this.doc.text(label, 20, yPos)
      this.doc.setFont(undefined, 'normal')
      this.doc.text(value, 100, yPos)
      yPos += 8
    })
  }

  addHealthOverview(allCattleData) {
    let yPos = 170
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.healthOverview.toUpperCase(), 20, yPos)
    yPos += 15
    
    // Categorize cattle by health scores from actual data
    const excellentHealth = allCattleData.filter(cattle => 
      parseFloat(cattle.healthScore) >= 90).length
    const goodHealth = allCattleData.filter(cattle => {
      const score = parseFloat(cattle.healthScore)
      return score >= 80 && score < 90
    }).length
    const fairHealth = allCattleData.filter(cattle => {
      const score = parseFloat(cattle.healthScore)
      return score >= 60 && score < 80
    }).length
    const poorHealth = allCattleData.filter(cattle => 
      parseFloat(cattle.healthScore) < 60).length
    
    const avgHealthScore = allCattleData.length > 0 ? 
      allCattleData.reduce((sum, cattle) => sum + (parseFloat(cattle.healthScore) || 0), 0) / allCattleData.length : 0
    
    // Count disease predictions from AI model
    const healthyCattle = allCattleData.filter(cattle => 
      cattle.diseaseStatus === 'Healthy' || cattle.diseaseStatus === 'No Disease').length
    const diseasedCattle = allCattleData.length - healthyCattle
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const healthData = [
      [`${this.t.excellentHealth}: ${excellentHealth}`, `${this.t.goodHealth}: ${goodHealth}`],
      [`${this.t.fairHealth}: ${fairHealth}`, `${this.t.poorHealth}: ${poorHealth}`],
      [`${this.t.overallHealthRate}: ${avgHealthScore.toFixed(1)}/100`, `${this.t.healthyCattle}: ${healthyCattle}`]
    ]
    
    healthData.forEach(([label, value]) => {
      this.doc.setFont(undefined, 'bold')
      this.doc.text(label, 20, yPos)
      this.doc.setFont(undefined, 'normal')
      this.doc.text(value, 100, yPos)
      yPos += 8
    })
  }

  addCattleSummaryTable(allCattleData) {
    this.doc.addPage()
    let yPos = 30
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.individualCattleSummary.toUpperCase(), 20, yPos)
    yPos += 20
    
    // Table headers
    this.doc.setFontSize(10)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.cattleId, 20, yPos)
    this.doc.text(this.t.breed, 50, yPos)
    this.doc.text(this.t.age, 90, yPos)
    this.doc.text(this.t.dailyYield, 120, yPos)
    this.doc.text(this.t.healthScore, 160, yPos)
    this.doc.text(this.t.status, 190, yPos)
    yPos += 10
    
    // Draw line under headers
    this.doc.setLineWidth(0.3)
    this.doc.line(20, yPos - 2, 210, yPos - 2)
    yPos += 5
    
    // Table rows - using actual cattle data
    this.doc.setFont(undefined, 'normal')
    allCattleData.forEach((cattle, index) => {
      if (yPos > 270) { // Start new page if needed
        this.doc.addPage()
        yPos = 30
      }
      
      const cattleId = cattle.cattle_id?.substring(0, 8) || `C${index + 1}`
      const breed = cattle.breed?.substring(0, 8) || 'N/A'
      const age = cattle.age_months ? `${Math.floor(cattle.age_months / 12)}y` : 'N/A'
      const milkYield = cattle.dailyYield ? `${parseFloat(cattle.dailyYield).toFixed(1)}L` : '0.0L'
      const health = cattle.healthScore ? parseFloat(cattle.healthScore).toFixed(0) : '0'
      const healthStatus = health >= 80 ? this.t.good : health >= 60 ? this.t.fair : this.t.poor
      
      this.doc.text(cattleId, 20, yPos)
      this.doc.text(breed, 50, yPos)
      this.doc.text(age, 90, yPos)
      this.doc.text(milkYield, 120, yPos)
      this.doc.text(health, 160, yPos)
      this.doc.text(healthStatus, 190, yPos)
      yPos += 8
    })
  }

  addFarmRecommendations(cattleData) {
    // Add new page for recommendations
    this.doc.addPage()
    let yPos = 30
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.farmRecommendations.toUpperCase(), 20, yPos)
    yPos += 15
    
    // Generate recommendations based on actual data
    const recommendations = this.generateRecommendations(cattleData)
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    recommendations.forEach(rec => {
      this.doc.text(rec, 20, yPos, { maxWidth: 170 })
      yPos += 8
    })
  }

  generateRecommendations(cattleData) {
    const recommendations = []
    
    if (cattleData.length === 0) {
      recommendations.push('• No cattle data available for recommendations.')
      return recommendations
    }
    
    // Calculate statistics
    const totalCattle = cattleData.length
    const avgYield = cattleData.reduce((sum, cattle) => sum + (cattle.dailyYield || 0), 0) / totalCattle
    const avgHealth = cattleData.reduce((sum, cattle) => sum + (cattle.healthScore || 0), 0) / totalCattle
    const lowProducers = cattleData.filter(cattle => (cattle.dailyYield || 0) < 15).length
    const unhealthyCattle = cattleData.filter(cattle => (cattle.healthScore || 0) < 70).length
    
    // Production recommendations
    if (avgYield < 20) {
      recommendations.push('• Consider improving feed quality and quantity to increase milk production.')
      recommendations.push('• Review breeding program to include higher-yielding breeds.')
    }
    
    if (lowProducers > totalCattle * 0.3) {
      recommendations.push('• 30%+ of cattle are low producers. Focus on nutrition and health management.')
    }
    
    // Health recommendations
    if (avgHealth < 80) {
      recommendations.push('• Overall herd health is below optimal. Implement regular health monitoring.')
    }
    
    if (unhealthyCattle > 0) {
      recommendations.push(`• ${unhealthyCattle} cattle need immediate health attention.`)
      recommendations.push('• Schedule veterinary checkups for cattle with health scores below 70.')
    }
    
    // General recommendations
    recommendations.push('• Maintain consistent feeding schedules and monitor feed quality.')
    recommendations.push('• Ensure adequate water supply and clean housing conditions.')
    recommendations.push('• Regular AI predictions help optimize farm management decisions.')
    
    return recommendations
  }

  addFooter() {
    const pageWidth = this.doc.internal.pageSize.width
    const pageHeight = this.doc.internal.pageSize.height
    
    this.doc.setFontSize(8)
    this.doc.setFont(undefined, 'italic')
    this.doc.text('This report is generated by AI-powered cattle monitoring system for informational purposes.', 
                  pageWidth / 2, pageHeight - 20, { align: 'center' })
    
    this.doc.setFont(undefined, 'normal')
    this.doc.text('© 2024 Dairy Cattle Monitoring System', pageWidth / 2, pageHeight - 10, { align: 'center' })
  }

  // Download PDF
  downloadPDF(filename = 'cattle-report.pdf') {
    if (this.doc) {
      this.doc.save(filename)
    }
  }

  // Get PDF as blob
  getPDFBlob() {
    if (this.doc) {
      return this.doc.output('blob')
    }
    return null
  }

  // Save report to database
  async saveReportToDatabase(userId, reportType, reportData, pdfBlob) {
    try {
      const { data, error } = await supabase
        .from('reports')
        .insert({
          user_id: userId,
          report_type: reportType,
          report_data: reportData,
          generated_at: new Date().toISOString()
        })
        .select()
      
      if (error) throw error
      return data[0]
    } catch (error) {
      console.error('Error saving report to database:', error)
      throw error
    }
  }
}

// Utility function to fetch cattle data from Supabase
export async function fetchCattleData(userId, cattleId) {
  try {
    const { data, error } = await supabase
      .from('predictions')
      .select('*')
      .eq('user_id', userId)
      .eq('cattle_id', cattleId)
      .order('created_at', { ascending: false })
    
    if (error) throw error
    
    // Group predictions by type
    const milkPredictions = data.filter(p => p.prediction_type === 'milk_yield')
    const diseasePredictions = data.filter(p => p.prediction_type === 'disease_detection')
    
    return {
      milkYield: milkPredictions[0]?.prediction_result || null,
      disease: diseasePredictions[0]?.prediction_result || null,
      allPredictions: data
    }
  } catch (error) {
    console.error('Error fetching cattle data:', error)
    return { milkYield: null, disease: null, allPredictions: [] }
  }
}

// Utility function to save prediction to database
export async function savePredictionToDatabase(userId, cattleId, predictionType, inputData, predictionResult) {
  try {
    const { data, error } = await supabase
      .from('predictions')
      .insert({
        user_id: userId,
        cattle_id: cattleId,
        prediction_type: predictionType,
        input_data: inputData,
        prediction_result: predictionResult,
        confidence: predictionResult.confidence || null
      })
      .select()
    
    if (error) throw error
    
    return data[0]
  } catch (error) {
    console.error('Error saving prediction to database:', error)
    throw error
  }
}
