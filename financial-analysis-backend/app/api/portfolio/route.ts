import { type NextRequest, NextResponse } from "next/server"
import { z } from "zod"
import jwt from "jsonwebtoken"

// Validation schemas
const createPortfolioSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional(),
  isPublic: z.boolean().optional().default(false),
})

const addTransactionSchema = z.object({
  portfolioId: z.string().uuid(),
  symbol: z.string().min(1),
  type: z.enum(["buy", "sell"]),
  quantity: z.number().positive(),
  price: z.number().positive(),
  fees: z.number().min(0).optional().default(0),
  executedAt: z.string().optional(),
})

const updatePositionSchema = z.object({
  portfolioId: z.string().uuid(),
  symbol: z.string().min(1),
  quantity: z.number(),
  averageCost: z.number().positive(),
})

// Mock database
const portfolios = new Map()
const transactions = new Map()
const positions = new Map()

// Helper function to verify JWT token
function verifyToken(authHeader: string | null) {
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new Error("No token provided")
  }

  const token = authHeader.substring(7)
  return jwt.verify(token, process.env.JWT_SECRET || "fallback-secret") as any
}

// Helper function to calculate portfolio metrics
function calculatePortfolioMetrics(portfolioId: string) {
  const portfolioPositions = Array.from(positions.values()).filter((pos: any) => pos.portfolioId === portfolioId)

  let totalValue = 0
  let totalCost = 0
  let totalPnL = 0

  portfolioPositions.forEach((position: any) => {
    const marketValue = position.quantity * position.currentPrice
    const costBasis = position.quantity * position.averageCost

    totalValue += marketValue
    totalCost += costBasis
    totalPnL += marketValue - costBasis
  })

  const totalPnLPercent = totalCost > 0 ? (totalPnL / totalCost) * 100 : 0

  return {
    totalValue: Number(totalValue.toFixed(2)),
    totalCost: Number(totalCost.toFixed(2)),
    totalPnL: Number(totalPnL.toFixed(2)),
    totalPnLPercent: Number(totalPnLPercent.toFixed(2)),
    positionCount: portfolioPositions.length,
  }
}

// Helper function to update position after transaction
function updatePositionAfterTransaction(
  portfolioId: string,
  symbol: string,
  type: string,
  quantity: number,
  price: number,
) {
  const positionKey = `${portfolioId}-${symbol}`
  let position = positions.get(positionKey)

  if (!position) {
    position = {
      id: crypto.randomUUID(),
      portfolioId,
      symbol,
      quantity: 0,
      averageCost: 0,
      currentPrice: price, // Mock current price
    }
  }

  if (type === "buy") {
    const newTotalCost = position.quantity * position.averageCost + quantity * price
    const newQuantity = position.quantity + quantity
    position.averageCost = newTotalCost / newQuantity
    position.quantity = newQuantity
  } else if (type === "sell") {
    position.quantity = Math.max(0, position.quantity - quantity)
    if (position.quantity === 0) {
      position.averageCost = 0
    }
  }

  // Update current price with some random variation
  position.currentPrice = price * (1 + (Math.random() - 0.5) * 0.05)
  position.marketValue = position.quantity * position.currentPrice
  position.unrealizedPnL = position.marketValue - position.quantity * position.averageCost
  position.unrealizedPnLPercent =
    position.averageCost > 0 ? (position.unrealizedPnL / (position.quantity * position.averageCost)) * 100 : 0

  positions.set(positionKey, position)
  return position
}

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get("authorization")
    const decoded = verifyToken(authHeader)
    const { searchParams } = new URL(request.url)
    const portfolioId = searchParams.get("portfolioId")

    if (portfolioId) {
      // Get specific portfolio with positions
      const portfolio = portfolios.get(portfolioId)
      if (!portfolio || portfolio.userId !== decoded.userId) {
        return NextResponse.json({ success: false, error: "Portfolio not found" }, { status: 404 })
      }

      const portfolioPositions = Array.from(positions.values()).filter((pos: any) => pos.portfolioId === portfolioId)

      const portfolioTransactions = Array.from(transactions.values())
        .filter((tx: any) => tx.portfolioId === portfolioId)
        .sort((a: any, b: any) => new Date(b.executedAt).getTime() - new Date(a.executedAt).getTime())

      const metrics = calculatePortfolioMetrics(portfolioId)

      return NextResponse.json({
        success: true,
        data: {
          ...portfolio,
          ...metrics,
          positions: portfolioPositions,
          recentTransactions: portfolioTransactions.slice(0, 10),
        },
      })
    }

    // Get all portfolios for user
    const userPortfolios = Array.from(portfolios.values()).filter(
      (portfolio: any) => portfolio.userId === decoded.userId,
    )

    const portfoliosWithMetrics = userPortfolios.map((portfolio: any) => ({
      ...portfolio,
      ...calculatePortfolioMetrics(portfolio.id),
    }))

    return NextResponse.json({
      success: true,
      data: portfoliosWithMetrics,
    })
  } catch (error) {
    console.error("Portfolio GET error:", error)
    return NextResponse.json({ success: false, error: "Unauthorized" }, { status: 401 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get("authorization")
    const decoded = verifyToken(authHeader)
    const body = await request.json()
    const { action } = body

    if (action === "create") {
      const { name, description, isPublic } = createPortfolioSchema.parse(body)

      const portfolio = {
        id: crypto.randomUUID(),
        userId: decoded.userId,
        name,
        description: description || "",
        isPublic: isPublic || false,
        isDefault: false,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }

      portfolios.set(portfolio.id, portfolio)

      return NextResponse.json({
        success: true,
        data: portfolio,
      })
    }

    if (action === "add-transaction") {
      const { portfolioId, symbol, type, quantity, price, fees, executedAt } = addTransactionSchema.parse(body)

      // Verify portfolio ownership
      const portfolio = portfolios.get(portfolioId)
      if (!portfolio || portfolio.userId !== decoded.userId) {
        return NextResponse.json({ success: false, error: "Portfolio not found" }, { status: 404 })
      }

      const transaction = {
        id: crypto.randomUUID(),
        portfolioId,
        symbol,
        type,
        quantity,
        price,
        fees: fees || 0,
        totalAmount: type === "buy" ? quantity * price + (fees || 0) : quantity * price - (fees || 0),
        executedAt: executedAt || new Date().toISOString(),
        createdAt: new Date().toISOString(),
      }

      transactions.set(transaction.id, transaction)

      // Update position
      const updatedPosition = updatePositionAfterTransaction(portfolioId, symbol, type, quantity, price)

      return NextResponse.json({
        success: true,
        data: {
          transaction,
          position: updatedPosition,
        },
      })
    }

    if (action === "update-position") {
      const { portfolioId, symbol, quantity, averageCost } = updatePositionSchema.parse(body)

      // Verify portfolio ownership
      const portfolio = portfolios.get(portfolioId)
      if (!portfolio || portfolio.userId !== decoded.userId) {
        return NextResponse.json({ success: false, error: "Portfolio not found" }, { status: 404 })
      }

      const positionKey = `${portfolioId}-${symbol}`
      let position = positions.get(positionKey)

      if (!position) {
        position = {
          id: crypto.randomUUID(),
          portfolioId,
          symbol,
        }
      }

      position.quantity = quantity
      position.averageCost = averageCost
      position.currentPrice = averageCost * (1 + (Math.random() - 0.5) * 0.1) // Mock current price
      position.marketValue = position.quantity * position.currentPrice
      position.unrealizedPnL = position.marketValue - position.quantity * position.averageCost
      position.unrealizedPnLPercent =
        position.averageCost > 0 ? (position.unrealizedPnL / (position.quantity * position.averageCost)) * 100 : 0

      positions.set(positionKey, position)

      return NextResponse.json({
        success: true,
        data: position,
      })
    }

    return NextResponse.json({ success: false, error: "Invalid action" }, { status: 400 })
  } catch (error) {
    console.error("Portfolio POST error:", error)

    if (error instanceof z.ZodError) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid input",
          details: error.errors,
        },
        { status: 400 },
      )
    }

    return NextResponse.json({ success: false, error: "Unauthorized" }, { status: 401 })
  }
}

export async function PUT(request: NextRequest) {
  try {
    const authHeader = request.headers.get("authorization")
    const decoded = verifyToken(authHeader)
    const body = await request.json()
    const { portfolioId, name, description, isPublic } = body

    const portfolio = portfolios.get(portfolioId)
    if (!portfolio || portfolio.userId !== decoded.userId) {
      return NextResponse.json({ success: false, error: "Portfolio not found" }, { status: 404 })
    }

    // Update portfolio
    portfolio.name = name || portfolio.name
    portfolio.description = description !== undefined ? description : portfolio.description
    portfolio.isPublic = isPublic !== undefined ? isPublic : portfolio.isPublic
    portfolio.updatedAt = new Date().toISOString()

    portfolios.set(portfolioId, portfolio)

    return NextResponse.json({
      success: true,
      data: portfolio,
    })
  } catch (error) {
    console.error("Portfolio PUT error:", error)
    return NextResponse.json({ success: false, error: "Unauthorized" }, { status: 401 })
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const authHeader = request.headers.get("authorization")
    const decoded = verifyToken(authHeader)
    const { searchParams } = new URL(request.url)
    const portfolioId = searchParams.get("portfolioId")

    if (!portfolioId) {
      return NextResponse.json({ success: false, error: "Portfolio ID required" }, { status: 400 })
    }

    const portfolio = portfolios.get(portfolioId)
    if (!portfolio || portfolio.userId !== decoded.userId) {
      return NextResponse.json({ success: false, error: "Portfolio not found" }, { status: 404 })
    }

    // Delete portfolio and related data
    portfolios.delete(portfolioId)

    // Delete positions
    Array.from(positions.keys()).forEach((key) => {
      if (key.startsWith(`${portfolioId}-`)) {
        positions.delete(key)
      }
    })

    // Delete transactions
    Array.from(transactions.values()).forEach((tx: any) => {
      if (tx.portfolioId === portfolioId) {
        transactions.delete(tx.id)
      }
    })

    return NextResponse.json({
      success: true,
      message: "Portfolio deleted successfully",
    })
  } catch (error) {
    console.error("Portfolio DELETE error:", error)
    return NextResponse.json({ success: false, error: "Unauthorized" }, { status: 401 })
  }
}
