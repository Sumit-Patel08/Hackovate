import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { Cog as Cow, Calendar, Milk, Activity, FileText, Download } from "lucide-react"
import { useState } from "react"
import { ReportGenerator, fetchCattleData } from "@/lib/reportGenerator"
import { useAuth } from "@/contexts/AuthContext"

export function CattleCard({ cattle, language = "en", userInputData = null }) {
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)
  const { user } = useAuth()
  const translations = {
    en: {
      cow: "Cow",
      age: "Age",
      years: "years",
      breed: "Breed",
      dailyYield: "Daily Yield",
      liters: "L",
      healthScore: "Health Score",
      lastCheckup: "Last Checkup",
      daysAgo: "days ago",
      report: "Report",
    },
    hi: {
      cow: "गाय",
      age: "उम्र",
      years: "वर्ष",
      breed: "नस्ल",
      dailyYield: "दैनिक उत्पादन",
      liters: "लीटर",
      healthScore: "स्वास्थ्य स्कोर",
      lastCheckup: "अंतिम जांच",
      daysAgo: "दिन पहले",
      report: "रिपोर्ट",
    },
    gu: {
      cow: "ગાય",
      age: "ઉંમર",
      years: "વર્ષ",
      breed: "જાતિ",
      dailyYield: "દૈનિક ઉત્પાદન",
      liters: "લીટર",
      healthScore: "સ્વાસ્થ્ય સ્કોર",
      lastCheckup: "છેલ્લી તપાસ",
      daysAgo: "દિવસ પહેલાં",
      report: "રિપોર્ટ",
    },
    mr: {
      cow: "गाय",
      age: "वय",
      years: "वर्षे",
      breed: "जात",
      dailyYield: "दैनिक उत्पादन",
      liters: "लिटर",
      healthScore: "आरोग्य स्कोर",
      lastCheckup: "शेवटची तपासणी",
      daysAgo: "दिवसांपूर्वी",
      report: "अहवाल",
    },
  }

  const t = translations[language]

  const handleGenerateReport = async () => {
    if (!user) {
      alert('Please log in to generate reports')
      return
    }

    setIsGeneratingReport(true)
    
    try {
      console.log('Starting individual cattle report generation...')
      console.log('Cattle data for report:', cattle)
      console.log('User ID:', user.id)
      
      // Create report generator instance with language (same as dashboard)
      const reportGenerator = new ReportGenerator(language)
      console.log('ReportGenerator created successfully')
      
      // Fetch latest predictions from Supabase
      const predictions = await fetchCattleData(user.id, cattle.id)
      console.log('Fetched predictions:', predictions)
      
      // Generate PDF report using the same pattern as dashboard
      const pdfDoc = await reportGenerator.generatePDFReport(
        cattle,
        predictions,
        userInputData
      )
      console.log('Individual cattle report generated successfully')
      
      // Generate filename with current date
      const currentDate = new Date().toISOString().split('T')[0]
      const filename = `cattle-${cattle.id}-report-${currentDate}.pdf`
      console.log('Generated filename:', filename)
      
      // Download the PDF
      reportGenerator.downloadPDF(filename)
      console.log('PDF download initiated')
      
      // Save report to database (optional - don't fail if this errors)
      try {
        const reportData = {
          cattle_info: cattle,
          predictions: predictions,
          user_input: userInputData,
          generated_at: new Date().toISOString()
        }
        
        await reportGenerator.saveReportToDatabase(
          user.id,
          `cattle_${cattle.id}`,
          reportData,
          reportGenerator.getPDFBlob()
        )
        console.log('Report metadata saved to database')
      } catch (dbError) {
        console.error('Error saving report metadata (non-critical):', dbError)
        // Don't show error to user as the PDF was still generated successfully
      }
      
      alert(`Individual cattle report generated successfully! Downloaded as ${filename}`)
      
    } catch (error) {
      console.error('=== INDIVIDUAL CATTLE REPORT ERROR ===')
      console.error('Full error object:', error)
      console.error('Error message:', error.message)
      console.error('Error stack:', error.stack)
      console.error('Cattle data:', cattle)
      console.error('User object:', user)
      
      // More specific error messages (same as dashboard)
      if (error.message?.includes('jsPDF')) {
        alert('PDF generation library error. Please refresh the page and try again.')
      } else if (error.message?.includes('No cattle data')) {
        alert('No cattle data available for report. Please try again.')
      } else if (error.message?.includes('translations')) {
        alert('Translation error. Trying with default language...')
        // Retry with English
        try {
          const reportGenerator = new ReportGenerator('en')
          const pdfDoc = await reportGenerator.generatePDFReport(cattle, predictions, userInputData)
          const filename = `cattle-${cattle.id}-report-${new Date().toISOString().split('T')[0]}.pdf`
          reportGenerator.downloadPDF(filename)
          alert(`Report generated successfully in English! Downloaded as ${filename}`)
        } catch (retryError) {
          console.error('Retry with English also failed:', retryError)
          alert('Failed to generate report. Please check console for details.')
        }
      } else {
        alert(`Failed to generate report: ${error.message}. Please check console for details.`)
      }
    } finally {
      setIsGeneratingReport(false)
    }
  }

  const getHealthColor = (score) => {
    if (score >= 80) return "bg-green-100 text-green-800"
    if (score >= 60) return "bg-yellow-100 text-yellow-800"
    return "bg-red-100 text-red-800"
  }

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Cow className="h-5 w-5 text-primary" />
            <span>
              {t.cow} #{cattle.id}
            </span>
          </div>
          <Badge variant="outline">{cattle.breed}</Badge>
        </CardTitle>
        <CardDescription className="flex items-center space-x-4">
          <span className="flex items-center space-x-1">
            <Calendar className="h-4 w-4" />
            <span>
              {cattle.age} {t.years}
            </span>
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Milk className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">{t.dailyYield}</span>
          </div>
          <span className="font-semibold">
            {cattle.dailyYield} {t.liters}
          </span>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">{t.healthScore}</span>
            </div>
            <Badge className={getHealthColor(cattle.healthScore)}>{cattle.healthScore}%</Badge>
          </div>
          <Progress value={cattle.healthScore} className="h-2" />
        </div>

        <div className="text-xs text-muted-foreground">
          {t.lastCheckup}: {cattle.lastCheckup} {t.daysAgo}
        </div>

        <div className="pt-3 border-t">
          <Button 
            variant="default" 
            size="sm" 
            className="w-full bg-green-600 hover:bg-green-700 text-white"
            onClick={handleGenerateReport}
            disabled={isGeneratingReport}
          >
            {isGeneratingReport ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Generating...
              </>
            ) : (
              <>
                <FileText className="h-4 w-4 mr-2" />
                {t.report}
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
