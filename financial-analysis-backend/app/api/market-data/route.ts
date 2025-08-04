import { type NextRequest, NextResponse } from "next/server"
import { z } from "zod"

// Validation schema
const marketDataSchema = z.object({
  symbols: z.array(z.string()).optional(),
  timeframe: z.enum(["1m", "5m", "15m", "1h", "4h", "1d"]).optional().default("1d"),
  limit: z.number().min(1).max(1000).optional().default(100),
  from: z.string().optional(),
  to: z.string().optional(),
})

// Mock market data generator
function generateMockMarketData(symbol: string, timeframe: string, limit: number) {
  const data = []
  const now = new Date()
  const timeframeMs =
    {
      "1m": 60 * 1000,
      "5m": 5 * 60 * 1000,
      "15m": 15 * 60 * 1000,
      "1h": 60 * 60 * 1000,
      "4h": 4 * 60 * 60 * 1000,
      "1d": 24 * 60 * 60 * 1000,
    }[timeframe] || 24 * 60 * 60 * 1000

  let basePrice = Math.random() * 1000 + 100 // Random base price between 100-1100

  for (let i = limit - 1; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * timeframeMs)

    // Generate realistic price movement
    const change = (Math.random() - 0.5) * 0.05 // ±2.5% change
    const open = basePrice
    const close = basePrice * (1 + change)
    const high = Math.max(open, close) * (1 + Math.random() * 0.02)
    const low = Math.min(open, close) * (1 - Math.random() * 0.02)
    const volume = Math.random() * 1000000 + 100000

    data.push({
      symbol,
      timestamp: timestamp.toISOString(),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Math.floor(volume),
      timeframe,
    })

    basePrice = close
  }

  return data.reverse()
}

// Mock real-time price data
function generateMockPriceData(symbol: string) {
  const basePrice = Math.random() * 1000 + 100
  const change24h = (Math.random() - 0.5) * 0.1 // ±5% change
  const price = basePrice * (1 + change24h)

  return {
    symbol,
    price: Number(price.toFixed(2)),
    change24h: Number((price - basePrice).toFixed(2)),
    changePercent24h: Number((change24h * 100).toFixed(2)),
    volume24h: Math.floor(Math.random() * 10000000 + 1000000),
    marketCap: Math.floor(price * (Math.random() * 1000000000 + 100000000)),
    lastUpdated: new Date().toISOString(),
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const queryParams = {
      symbols: searchParams.get("symbols")?.split(",") || ["AAPL", "MSFT", "GOOGL", "TSLA"],
      timeframe: searchParams.get("timeframe") || "1d",
      limit: Number.parseInt(searchParams.get("limit") || "100"),
      from: searchParams.get("from"),
      to: searchParams.get("to"),
    }

    const validatedParams = marketDataSchema.parse(queryParams)
    const { symbols, timeframe, limit } = validatedParams

    // Check if requesting historical data or real-time prices
    const dataType = searchParams.get("type") || "historical"

    if (dataType === "realtime") {
      // Return real-time price data
      const priceData = symbols.map((symbol) => generateMockPriceData(symbol))

      return NextResponse.json({
        success: true,
        data: priceData,
        timestamp: new Date().toISOString(),
        type: "realtime",
      })
    }

    // Return historical market data
    const marketData = symbols.reduce(
      (acc, symbol) => {
        acc[symbol] = generateMockMarketData(symbol, timeframe, limit)
        return acc
      },
      {} as Record<string, any[]>,
    )

    return NextResponse.json({
      success: true,
      data: marketData,
      metadata: {
        symbols,
        timeframe,
        limit,
        from: validatedParams.from,
        to: validatedParams.to,
        recordCount: Object.values(marketData).reduce((sum, data) => sum + data.length, 0),
      },
      timestamp: new Date().toISOString(),
      type: "historical",
    })
  } catch (error) {
    console.error("Market data error:", error)

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
    const { action, symbols, alerts } = body

    if (action === "subscribe") {
      // Mock WebSocket subscription setup
      return NextResponse.json({
        success: true,
        message: "Subscribed to real-time updates",
        symbols,
        subscriptionId: crypto.randomUUID(),
      })
    }

    if (action === "set-alerts") {
      // Mock alert setup
      return NextResponse.json({
        success: true,
        message: "Price alerts configured",
        alerts: alerts.map((alert: any) => ({
          ...alert,
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
        })),
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
    console.error("Market data POST error:", error)
    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
      },
      { status: 500 },
    )
  }
}
