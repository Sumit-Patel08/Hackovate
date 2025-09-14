import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'
import { supabase } from './supabase'
import { translations } from './translations'

export class ReportGenerator {
  constructor(language = 'en') {
    this.doc = null
    this.language = language
    this.t = translations[language] || translations.en
  }

  // Generate comprehensive PDF report
  async generatePDFReport(cattleData, predictions, userInput) {
    this.doc = new jsPDF()
    
    // Set up document properties
    this.doc.setProperties({
      title: `${this.t.cattleHealthMilkReport} - ${cattleData.cattle_id}`,
      subject: this.t.comprehensiveCattleReport,
      author: this.t.dairyCattleMonitoring,
      creator: this.t.aiPoweredPlatform
    })

    // Add header
    this.addHeader(cattleData)
    
    // Add cattle information
    this.addCattleInfo(cattleData, userInput)
    
    // Add predictions section
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
    this.doc.text(this.t.cattleHealthMilkReport.toUpperCase(), pageWidth / 2, 20, { align: 'center' })
    
    // Subtitle
    this.doc.setFontSize(14)
    this.doc.setFont(undefined, 'normal')
    this.doc.text(`${this.t.cattleId}: ${cattleData.cattle_id || 'N/A'}`, pageWidth / 2, 30, { align: 'center' })
    
    // Date
    this.doc.setFontSize(10)
    this.doc.text(`${this.t.generatedOn}: ${new Date().toLocaleDateString('en-US', { 
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
    this.doc.text(this.t.cattleInformation.toUpperCase(), 20, yPos)
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
    this.doc.text(this.t.aiPredictionsAnalysis.toUpperCase(), 20, yPos)
    yPos += 15
    
    // Milk Yield Prediction - Use actual prediction data from Supabase
    if (predictions?.milkYield || predictions?.milkPrediction) {
      const milkData = predictions.milkYield || predictions.milkPrediction?.prediction_result
      
      this.doc.setFontSize(14)
      this.doc.setFont(undefined, 'bold')
      this.doc.text(this.t.milkYieldPrediction, 20, yPos)
      yPos += 10
      
      this.doc.setFontSize(11)
      this.doc.setFont(undefined, 'normal')
      this.doc.text(`${this.t.predictedDailyYield}: ${milkData?.predicted_milk_yield || milkData?.prediction || 'N/A'} liters`, 25, yPos)
      yPos += 8
      this.doc.text(`${this.t.confidenceLevel}: ${(milkData?.confidence * 100)?.toFixed(1) || 'N/A'}%`, 25, yPos)
      yPos += 8
      this.doc.text(`${this.t.timestamp}: ${predictions.milkPrediction?.created_at ? new Date(predictions.milkPrediction.created_at).toLocaleString() : 'N/A'}`, 25, yPos)
      yPos += 15
    }
    
    // Disease Detection Analysis - Use actual disease prediction data
    if (predictions?.disease || predictions?.diseasePrediction) {
      const diseaseData = predictions.disease || predictions.diseasePrediction?.prediction_result
      
      this.doc.setFontSize(14)
      this.doc.setFont(undefined, 'bold')
      this.doc.text(this.t.diseaseDetectionResults, 20, yPos)
      yPos += 10
      
      this.doc.setFontSize(11)
      this.doc.setFont(undefined, 'normal')
      this.doc.text(`${this.t.predictedDisease}: ${diseaseData?.predicted_disease || 'N/A'}`, 25, yPos)
      yPos += 8
      this.doc.text(`${this.t.riskLevel}: ${diseaseData?.risk_level || 'N/A'}`, 25, yPos)
      yPos += 8
      this.doc.text(`${this.t.confidence}: ${(diseaseData?.confidence * 100)?.toFixed(1) || 'N/A'}%`, 25, yPos)
      yPos += 8
      this.doc.text(`${this.t.timestamp}: ${predictions.diseasePrediction?.created_at ? new Date(predictions.diseasePrediction.created_at).toLocaleString() : 'N/A'}`, 25, yPos)
      yPos += 15
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
    
    // Add milk yield recommendations from actual predictions
    const milkData = predictions?.milkYield || predictions?.milkPrediction?.prediction_result
    if (milkData?.predicted_milk_yield) {
      const milkYield = parseFloat(milkData.predicted_milk_yield)
      if (milkYield < 20) {
        recommendations.push('• Consider improving feed quality and quantity to increase milk yield')
        recommendations.push('• Monitor for potential health issues affecting milk production')
        recommendations.push('• Review nutritional requirements and feeding schedule')
      } else if (milkYield > 30) {
        recommendations.push('• Excellent milk production! Maintain current feeding and care routine')
        recommendations.push('• Monitor for signs of metabolic stress due to high production')
        recommendations.push('• Ensure adequate calcium and energy supplementation')
      } else {
        recommendations.push('• Good milk production levels, continue current management')
        recommendations.push('• Consider gradual improvements in feed quality')
      }
    }
    
    // Add disease recommendations from actual predictions
    const diseaseData = predictions?.disease || predictions?.diseasePrediction?.prediction_result
    if (diseaseData?.recommendations && Array.isArray(diseaseData.recommendations)) {
      diseaseData.recommendations.forEach(rec => {
        recommendations.push(`• ${rec}`)
      })
    } else if (diseaseData?.predicted_disease && diseaseData.predicted_disease !== 'healthy') {
      const riskLevel = diseaseData.risk_level
      if (riskLevel === 'high') {
        recommendations.push('• Immediate veterinary consultation recommended')
        recommendations.push('• Isolate animal and monitor closely')
        recommendations.push('• Review biosecurity measures')
      } else if (riskLevel === 'medium') {
        recommendations.push('• Schedule veterinary checkup within 24-48 hours')
        recommendations.push('• Monitor symptoms and behavior changes')
        recommendations.push('• Ensure proper hygiene and sanitation')
      } else {
        recommendations.push('• Continue regular monitoring')
        recommendations.push('• Maintain preventive health measures')
      }
    }
    
    // General recommendations
    if (recommendations.length === 0) {
      recommendations.push('• Regular veterinary checkups recommended every 3-6 months')
      recommendations.push('• Monitor daily feed intake and water consumption')
      recommendations.push('• Maintain clean and comfortable housing conditions')
      recommendations.push('• Keep detailed records of health and production data')
    }
    
    recommendations.forEach(rec => {
      const lines = this.doc.splitTextToSize(rec, 170)
      lines.forEach(line => {
        this.doc.text(line, 20, yPos)
        yPos += 6
      })
      yPos += 2
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
  async saveReportToDatabase(userId, cattleId, reportData, pdfBlob) {
    try {
      // Upload PDF to Supabase Storage (if storage is configured)
      let pdfUrl = null
      
      // Insert report metadata into database
      const { data, error } = await supabase
        .from('reports')
        .insert({
          user_id: userId,
          cattle_id: cattleId,
          report_type: 'comprehensive',
          report_data: reportData,
          pdf_url: pdfUrl
        })
        .select()
      
      if (error) throw error
      
      return data[0]
    } catch (error) {
      console.error('Error saving report to database:', error)
      throw error
    }
  }

  // Generate comprehensive dashboard report for all cattle
  async generateDashboardReport(allCattleData, userId) {
    this.doc = new jsPDF()
    
    // Set up document properties
    this.doc.setProperties({
      title: `${this.t.farmDashboardReport} - ${new Date().toLocaleDateString()}`,
      subject: this.t.completeFarmAnalysis,
      author: this.t.dairyCattleMonitoring,
      creator: this.t.aiPoweredPlatform
    })

    // Add header
    this.addDashboardHeader(allCattleData.length)
    
    // Add farm summary
    this.addFarmSummary(allCattleData)
    
    // Add production analysis
    this.addProductionAnalysis(allCattleData)
    
    // Add health overview
    this.addHealthOverview(allCattleData)
    
    // Add individual cattle summary
    this.addCattleSummaryTable(allCattleData)
    
    // Add recommendations
    this.addFarmRecommendations(allCattleData)
    
    // Add footer
    this.addFooter()
    
    return this.doc
  }

  addDashboardHeader(totalCattle) {
    const pageWidth = this.doc.internal.pageSize.width
    
    // Title
    this.doc.setFontSize(20)
    this.doc.setFont(undefined, 'bold')
    this.doc.text(this.t.farmDashboardReport.toUpperCase(), pageWidth / 2, 20, { align: 'center' })
    
    // Subtitle
    this.doc.setFontSize(14)
    this.doc.setFont(undefined, 'normal')
    this.doc.text(`${this.t.completeFarmAnalysis} - ${totalCattle} ${this.t.cattle}`, pageWidth / 2, 30, { align: 'center' })
    
    // Date
    this.doc.setFontSize(10)
    this.doc.text(`${this.t.generatedOn}: ${new Date().toLocaleDateString('en-US', { 
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
    const totalMilkProduction = allCattleData.reduce((sum, cattle) => 
      sum + (parseFloat(cattle.predicted_milk_yield) || 0), 0)
    const avgYieldPerCattle = totalMilkProduction / totalCattle
    const projectedMonthly = totalMilkProduction * 30
    
    // Get most common breed
    const breedCounts = {}
    allCattleData.forEach(cattle => {
      const breed = cattle.breed || 'Unknown'
      breedCounts[breed] = (breedCounts[breed] || 0) + 1
    })
    const mostCommonBreed = Object.keys(breedCounts).reduce((a, b) => 
      breedCounts[a] > breedCounts[b] ? a : b)
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const summaryData = [
      [`${this.t.totalCattle}: ${totalCattle}`, `${this.t.dailyMilkProduction}: ${totalMilkProduction.toFixed(1)}L`],
      [`${this.t.averageYieldPerCattle}: ${avgYieldPerCattle.toFixed(1)}L`, `${this.t.projectedMonthlyProduction}: ${projectedMonthly.toFixed(0)}L`],
      [`${this.t.mostCommonBreed}: ${mostCommonBreed}`, ``]
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
    
    // Categorize cattle by production levels
    const lowProducers = allCattleData.filter(cattle => 
      parseFloat(cattle.predicted_milk_yield) < 15).length
    const mediumProducers = allCattleData.filter(cattle => {
      const milkYield = parseFloat(cattle.predicted_milk_yield)
      return milkYield >= 15 && milkYield <= 25
    }).length
    const highProducers = allCattleData.filter(cattle => 
      parseFloat(cattle.predicted_milk_yield) > 25).length
    
    const efficiency = ((highProducers / allCattleData.length) * 100).toFixed(1)
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const productionData = [
      [`${this.t.lowProducers}: ${lowProducers}`, `${this.t.mediumProducers}: ${mediumProducers}`],
      [`${this.t.highProducers}: ${highProducers}`, `${this.t.productionEfficiency}: ${efficiency}${this.t.highProducersPercent}`]
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
    
    // Categorize cattle by health scores
    const excellentHealth = allCattleData.filter(cattle => 
      parseFloat(cattle.health_score) >= 90).length
    const goodHealth = allCattleData.filter(cattle => {
      const score = parseFloat(cattle.health_score)
      return score >= 80 && score < 90
    }).length
    const fairHealth = allCattleData.filter(cattle => {
      const score = parseFloat(cattle.health_score)
      return score >= 60 && score < 80
    }).length
    const poorHealth = allCattleData.filter(cattle => 
      parseFloat(cattle.health_score) < 60).length
    
    const avgHealthScore = allCattleData.reduce((sum, cattle) => 
      sum + (parseFloat(cattle.health_score) || 0), 0) / allCattleData.length
    
    this.doc.setFontSize(11)
    this.doc.setFont(undefined, 'normal')
    
    const healthData = [
      [`${this.t.excellentHealth}: ${excellentHealth}`, `${this.t.goodHealth}: ${goodHealth}`],
      [`${this.t.fairHealth}: ${fairHealth}`, `${this.t.poorHealth}: ${poorHealth}`],
      [`${this.t.overallHealthRate}: ${avgHealthScore.toFixed(1)}/100`, ``]
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
    this.doc.line(20, yPos - 2, 190, yPos - 2)
    yPos += 5
    
    // Table rows
    this.doc.setFont(undefined, 'normal')
    allCattleData.forEach((cattle, index) => {
      if (yPos > 270) { // Start new page if needed
        this.doc.addPage()
        yPos = 30
      }
      
      const cattleId = cattle.cattle_id?.substring(0, 8) || `C${index + 1}`
      const breed = cattle.breed?.substring(0, 8) || 'N/A'
      const age = cattle.age_months ? `${Math.floor(cattle.age_months / 12)}y` : 'N/A'
      const milkYield = cattle.predicted_milk_yield ? `${cattle.predicted_milk_yield.toFixed(1)}L` : '0.0L'
      const health = cattle.health_score ? cattle.health_score.toFixed(0) : '0'
      const healthStatus = health >= 80 ? this.t.good : health >= 60 ? this.t.fair : this.t.poor
      
      this.doc.text(cattleId, 20, yPos)
      this.doc.text(breed, 50, yPos)
      this.doc.text(age, 80, yPos)
      this.doc.text(milkYield, 100, yPos)
      this.doc.text(health, 160, yPos)
      this.doc.text(healthStatus, 190, yPos)
      this.doc.text(status, 160, yPos)
      yPos += 8
    })
  }

  addFarmRecommendations(cattleData) {
    // Add new page for recommendations
    this.doc.addPage()
    let yPos = 30
    
    this.doc.setFontSize(16)
    this.doc.setFont(undefined, 'bold')
    this.doc.text('FARM RECOMMENDATIONS', 20, yPos)
    yPos += 15
    
    this.doc.setFontSize(12)
    this.doc.setFont(undefined, 'normal')
    
    const recommendations = []
    
    // Production recommendations
    const totalDailyYield = cattleData.reduce((sum, cattle) => sum + (cattle.dailyYield || 0), 0)
    const avgYield = cattleData.length > 0 ? totalDailyYield / cattleData.length : 0
    
    if (avgYield < 20) {
      recommendations.push('• Consider improving overall feed quality and nutrition program')
      recommendations.push('• Review feeding schedules and ensure consistent feed supply')
    } else if (avgYield > 25) {
      recommendations.push('• Excellent production levels! Maintain current management practices')
      recommendations.push('• Monitor for metabolic disorders in high-producing cattle')
    }
    
    // Health recommendations
    const healthyCattle = cattleData.filter(cattle => (cattle.healthScore || 0) >= 80).length
    const healthRate = cattleData.length > 0 ? (healthyCattle / cattleData.length) * 100 : 0
    
    if (healthRate < 70) {
      recommendations.push('• Implement comprehensive health monitoring program')
      recommendations.push('• Schedule veterinary consultation for herd health assessment')
    }
    
    // General recommendations
    recommendations.push('• Maintain detailed production and health records for each animal')
    recommendations.push('• Implement regular body condition scoring')
    recommendations.push('• Ensure adequate water supply and quality')
    recommendations.push('• Review and optimize housing conditions for comfort')
    recommendations.push('• Consider genetic improvement programs for future breeding')
    
    recommendations.forEach(rec => {
      this.doc.text(rec, 20, yPos, { maxWidth: 170 })
      yPos += 8
    })
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
