import { type NextRequest, NextResponse } from "next/server"
import { z } from "zod"

// Validation schema
const predictionSchema = z.object({
  symbols: z.array(z.string()).min(1),
  timeframe: z.enum(["1h", "4h", "1d", "1w"]).optional().default("1d"),
  horizon: z.number().min(1).max(30).optional().default(1),
  modelType: z.enum(["lstm", "xgboost", "ensemble"]).optional().default("ensemble"),
})

// Mock ML models and predictions
const mockModels = {
  lstm: {
    name: "LSTM Neural Network",
    accuracy: 0.72,
    description: "Deep learning model for time series prediction",
  },
  xgboost: {
    name: "XGBoost Regressor",
    accuracy: 0.68,
    description: "Gradient boosting model for price direction",
  },
  ensemble: {
    name: "Ensemble Model",
    accuracy: 0.75,
    description: "Combined model using multiple algorithms",
  },
}

function generateMockPrediction(symbol: string, timeframe: string, horizon: number, modelType: string) {
  const currentPrice = Math.random() * 1000 + 100
  const volatility = Math.random() * 0.1 + 0.02 // 2-12% volatility

  // Generate prediction based on model type
  let priceChange = 0
  let confidence = 0

  switch (modelType) {
    case "lstm":
      priceChange = (Math.random() - 0.5) * volatility * 1.2
      confidence = 0.65 + Math.random() * 0.15
      break
    case "xgboost":
      priceChange = (Math.random() - 0.5) * volatility * 0.8
      confidence = 0.6 + Math.random() * 0.2
      break
    case "ensemble":
      priceChange = (Math.random() - 0.5) * volatility
      confidence = 0.7 + Math.random() * 0.15
      break
  }

  const predictedPrice = currentPrice * (1 + priceChange)
  const direction = priceChange > 0 ? "bullish" : "bearish"

  // Generate technical indicators
  const indicators = {
    rsi: Math.random() * 100,
    macd: (Math.random() - 0.5) * 10,
    bollinger_position: Math.random(),
    volume_trend: Math.random() > 0.5 ? "increasing" : "decreasing",
  }

  // Generate risk metrics
  const riskMetrics = {
    volatility: volatility * 100,
    var_5: currentPrice * volatility * -2.33, // 5% VaR
    sharpe_ratio: (Math.random() - 0.3) * 3,
    max_drawdown: -(Math.random() * 0.3 + 0.05),
  }

  // Generate market sentiment
  const sentiment = {
    news_sentiment: (Math.random() - 0.5) * 2,
    social_sentiment: (Math.random() - 0.5) * 2,
    fear_greed_index: Math.random() * 100,
  }

  return {
    symbol,
    currentPrice: Number(currentPrice.toFixed(2)),
    predictedPrice: Number(predictedPrice.toFixed(2)),
    priceChange: Number((predictedPrice - currentPrice).toFixed(2)),
    priceChangePercent: Number((priceChange * 100).toFixed(2)),
    direction,
    confidence: Number(confidence.toFixed(3)),
    timeframe,
    horizon,
    modelType,
    modelInfo: mockModels[modelType as keyof typeof mockModels],
    indicators,
    riskMetrics,
    sentiment,
    features: ["price_momentum", "volume_profile", "technical_indicators", "market_sentiment", "volatility_patterns"],
    timestamp: new Date().toISOString(),
    expiresAt: new Date(Date.now() + horizon * 24 * 60 * 60 * 1000).toISOString(),
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const queryParams = {
      symbols: searchParams.get("symbols")?.split(",") || ["AAPL"],
      timeframe: searchParams.get("timeframe") || "1d",
      horizon: Number.parseInt(searchParams.get("horizon") || "1"),
      modelType: searchParams.get("modelType") || "ensemble",
    }

    const validatedParams = predictionSchema.parse(queryParams)
    const { symbols, timeframe, horizon, modelType } = validatedParams

    // Generate predictions for each symbol
    const predictions = symbols.map((symbol) => generateMockPrediction(symbol, timeframe, horizon, modelType))

    // Calculate ensemble metrics
    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length
    const bullishCount = predictions.filter((p) => p.direction === "bullish").length
    const bearishCount = predictions.filter((p) => p.direction === "bearish").length

    return NextResponse.json({
      success: true,
      data: {
        predictions,
        summary: {
          totalSymbols: symbols.length,
          averageConfidence: Number(avgConfidence.toFixed(3)),
          bullishSignals: bullishCount,
          bearishSignals: bearishCount,
          marketSentiment:
            bullishCount > bearishCount ? "bullish" : bearishCount > bullishCount ? "bearish" : "neutral",
        },
        metadata: {
          timeframe,
          horizon,
          modelType,
          generatedAt: new Date().toISOString(),
        },
      },
    })
  } catch (error) {
    console.error("Predictions error:", error)

    if (error instanceof z.ZodError) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid parameters",
          details: error.errors,
        },
        { status: 400 },
      )
    }

    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
      },
      { status: 500 },
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, symbols, modelConfig } = body

    if (action === "train-model") {
      // Mock model training
      const trainingJob = {
        id: crypto.randomUUID(),
        symbols,
        modelType: modelConfig?.type || "ensemble",
        status: "training",
        progress: 0,
        startedAt: new Date().toISOString(),
        estimatedCompletion: new Date(Date.now() + 30 * 60 * 1000).toISOString(), // 30 minutes
      }

      // Simulate training progress
      setTimeout(() => {
        trainingJob.status = "completed"
        trainingJob.progress = 100
      }, 5000)

      return NextResponse.json({
        success: true,
        data: trainingJob,
      })
    }

    if (action === "backtest") {
      // Mock backtesting results
      const backtestResults = {
        id: crypto.randomUUID(),
        symbols,
        period: "1y",
        totalTrades: Math.floor(Math.random() * 100 + 50),
        winRate: Number((Math.random() * 0.3 + 0.5).toFixed(3)), // 50-80%
        avgReturn: Number((Math.random() * 0.2 - 0.05).toFixed(4)), // -5% to 15%
        sharpeRatio: Number((Math.random() * 2 - 0.5).toFixed(2)), // -0.5 to 1.5
        maxDrawdown: Number(-(Math.random() * 0.3 + 0.05).toFixed(3)), // -5% to -35%
        volatility: Number((Math.random() * 0.3 + 0.1).toFixed(3)), // 10-40%
        createdAt: new Date().toISOString(),
      }

      return NextResponse.json({
        success: true,
        data: backtestResults,
      })
    }

    return NextResponse.json(
      {
        success: false,
        error: "Invalid action",
      },
      { status: 400 },
    )
  } catch (error) {
    console.error("Predictions POST error:", error)
    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
      },
      { status: 500 },
    )
  }
}
