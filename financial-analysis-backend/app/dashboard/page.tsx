"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  AlertTriangle,
  RefreshCw,
  Eye,
  Plus,
  Settings,
} from "lucide-react"

interface MarketData {
  symbol: string
  price: number
  change24h: number
  changePercent24h: number
  volume24h: number
  marketCap: number
}

interface Portfolio {
  id: string
  name: string
  totalValue: number
  totalPnL: number
  totalPnLPercent: number
  positionCount: number
}

interface Prediction {
  symbol: string
  currentPrice: number
  predictedPrice: number
  priceChangePercent: number
  direction: "bullish" | "bearish"
  confidence: number
}

export default function Dashboard() {
  const [marketData, setMarketData] = useState<MarketData[]>([])
  const [portfolios, setPortfolios] = useState<Portfolio[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  // Mock data for demonstration
  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    setLoading(true)
    try {
      // Simulate API calls
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Mock market data
      setMarketData([
        {
          symbol: "AAPL",
          price: 185.92,
          change24h: 2.34,
          changePercent24h: 1.28,
          volume24h: 45678900,
          marketCap: 2890000000000,
        },
        {
          symbol: "MSFT",
          price: 378.85,
          change24h: -1.45,
          changePercent24h: -0.38,
          volume24h: 23456700,
          marketCap: 2810000000000,
        },
        {
          symbol: "GOOGL",
          price: 142.56,
          change24h: 3.21,
          changePercent24h: 2.3,
          volume24h: 34567800,
          marketCap: 1790000000000,
        },
        {
          symbol: "TSLA",
          price: 248.73,
          change24h: -5.67,
          changePercent24h: -2.23,
          volume24h: 67890100,
          marketCap: 789000000000,
        },
        {
          symbol: "BTC",
          price: 43250.0,
          change24h: 1250.0,
          changePercent24h: 2.98,
          volume24h: 15678900000,
          marketCap: 847000000000,
        },
        {
          symbol: "ETH",
          price: 2650.0,
          change24h: -45.0,
          changePercent24h: -1.67,
          volume24h: 8901200000,
          marketCap: 318000000000,
        },
      ])

      // Mock portfolio data
      setPortfolios([
        {
          id: "1",
          name: "Main Portfolio",
          totalValue: 125430.5,
          totalPnL: 8750.25,
          totalPnLPercent: 7.5,
          positionCount: 8,
        },
        {
          id: "2",
          name: "Crypto Holdings",
          totalValue: 45670.0,
          totalPnL: -2340.75,
          totalPnLPercent: -4.9,
          positionCount: 5,
        },
      ])

      // Mock predictions
      setPredictions([
        {
          symbol: "AAPL",
          currentPrice: 185.92,
          predictedPrice: 192.5,
          priceChangePercent: 3.54,
          direction: "bullish",
          confidence: 0.78,
        },
        {
          symbol: "TSLA",
          currentPrice: 248.73,
          predictedPrice: 235.2,
          priceChangePercent: -5.44,
          direction: "bearish",
          confidence: 0.72,
        },
        {
          symbol: "BTC",
          currentPrice: 43250.0,
          predictedPrice: 46800.0,
          priceChangePercent: 8.21,
          direction: "bullish",
          confidence: 0.65,
        },
      ])
    } catch (error) {
      console.error("Error loading dashboard data:", error)
    } finally {
      setLoading(false)
    }
  }

  const refreshData = async () => {
    setRefreshing(true)
    await loadDashboardData()
    setRefreshing(false)
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  const formatLargeNumber = (value: number) => {
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`
    if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`
    return formatCurrency(value)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="h-8 bg-slate-200 rounded w-1/4"></div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-32 bg-slate-200 rounded"></div>
              ))}
            </div>
            <div className="h-96 bg-slate-200 rounded"></div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
            <p className="text-slate-600">Monitor your investments and market insights</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={refreshData} disabled={refreshing}>
              <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
              Refresh
            </Button>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Asset
            </Button>
          </div>
        </div>

        {/* Portfolio Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Portfolio Value</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatCurrency(portfolios.reduce((sum, p) => sum + p.totalValue, 0))}
              </div>
              <p className="text-xs text-muted-foreground">+2.5% from last month</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {formatCurrency(portfolios.reduce((sum, p) => sum + p.totalPnL, 0))}
              </div>
              <p className="text-xs text-muted-foreground">+4.2% overall return</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Positions</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{portfolios.reduce((sum, p) => sum + p.positionCount, 0)}</div>
              <p className="text-xs text-muted-foreground">Across {portfolios.length} portfolios</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Alerts</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">3</div>
              <p className="text-xs text-muted-foreground">2 price alerts, 1 news alert</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="market" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="market">Market Overview</TabsTrigger>
            <TabsTrigger value="portfolios">Portfolios</TabsTrigger>
            <TabsTrigger value="predictions">AI Predictions</TabsTrigger>
            <TabsTrigger value="alerts">Alerts & News</TabsTrigger>
          </TabsList>

          {/* Market Overview */}
          <TabsContent value="market" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Market Overview</CardTitle>
                <CardDescription>Real-time market data and top movers</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {marketData.map((asset) => (
                    <div key={asset.symbol} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 bg-slate-100 rounded-full flex items-center justify-center">
                          <span className="font-semibold text-sm">{asset.symbol}</span>
                        </div>
                        <div>
                          <h3 className="font-semibold">{asset.symbol}</h3>
                          <p className="text-sm text-slate-600">Vol: {formatLargeNumber(asset.volume24h)}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{formatCurrency(asset.price)}</div>
                        <div
                          className={`flex items-center text-sm ${
                            asset.changePercent24h >= 0 ? "text-green-600" : "text-red-600"
                          }`}
                        >
                          {asset.changePercent24h >= 0 ? (
                            <TrendingUp className="h-3 w-3 mr-1" />
                          ) : (
                            <TrendingDown className="h-3 w-3 mr-1" />
                          )}
                          {asset.changePercent24h.toFixed(2)}%
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Portfolios */}
          <TabsContent value="portfolios" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {portfolios.map((portfolio) => (
                <Card key={portfolio.id}>
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle>{portfolio.name}</CardTitle>
                        <CardDescription>{portfolio.positionCount} positions</CardDescription>
                      </div>
                      <Button variant="outline" size="sm">
                        <Settings className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-600">Total Value</span>
                        <span className="font-semibold">{formatCurrency(portfolio.totalValue)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-600">P&L</span>
                        <div className={`font-semibold ${portfolio.totalPnL >= 0 ? "text-green-600" : "text-red-600"}`}>
                          {formatCurrency(portfolio.totalPnL)} ({portfolio.totalPnLPercent.toFixed(2)}%)
                        </div>
                      </div>
                      <div className="pt-2">
                        <Button className="w-full bg-transparent" variant="outline">
                          View Details
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* AI Predictions */}
          <TabsContent value="predictions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>AI Price Predictions</CardTitle>
                <CardDescription>Machine learning powered price forecasts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {predictions.map((prediction) => (
                    <div key={prediction.symbol} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 bg-slate-100 rounded-full flex items-center justify-center">
                          <span className="font-semibold text-sm">{prediction.symbol}</span>
                        </div>
                        <div>
                          <h3 className="font-semibold">{prediction.symbol}</h3>
                          <p className="text-sm text-slate-600">Current: {formatCurrency(prediction.currentPrice)}</p>
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold">{formatCurrency(prediction.predictedPrice)}</div>
                        <div
                          className={`text-sm ${
                            prediction.direction === "bullish" ? "text-green-600" : "text-red-600"
                          }`}
                        >
                          {prediction.priceChangePercent > 0 ? "+" : ""}
                          {prediction.priceChangePercent.toFixed(2)}%
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={prediction.direction === "bullish" ? "default" : "destructive"}>
                          {prediction.direction}
                        </Badge>
                        <p className="text-xs text-slate-600 mt-1">
                          {(prediction.confidence * 100).toFixed(0)}% confidence
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Alerts & News */}
          <TabsContent value="alerts" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Active Alerts</CardTitle>
                  <CardDescription>Price and portfolio alerts</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-yellow-50 border border-yellow-200 rounded">
                      <div>
                        <p className="font-medium">AAPL Price Alert</p>
                        <p className="text-sm text-slate-600">Target: $190.00</p>
                      </div>
                      <Badge variant="outline">Active</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded">
                      <div>
                        <p className="font-medium">Portfolio Loss Alert</p>
                        <p className="text-sm text-slate-600">Crypto Holdings -5%</p>
                      </div>
                      <Badge variant="destructive">Triggered</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded">
                      <div>
                        <p className="font-medium">BTC Price Target</p>
                        <p className="text-sm text-slate-600">Target: $45,000</p>
                      </div>
                      <Badge variant="default">Active</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Market News</CardTitle>
                  <CardDescription>Latest financial news and updates</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="p-3 border rounded">
                      <h4 className="font-medium">Apple Reports Strong Q4 Earnings</h4>
                      <p className="text-sm text-slate-600 mt-1">
                        Revenue beats expectations with strong iPhone sales...
                      </p>
                      <p className="text-xs text-slate-500 mt-2">2 hours ago</p>
                    </div>
                    <div className="p-3 border rounded">
                      <h4 className="font-medium">Bitcoin Surges Past $43K</h4>
                      <p className="text-sm text-slate-600 mt-1">
                        Cryptocurrency markets rally on institutional adoption...
                      </p>
                      <p className="text-xs text-slate-500 mt-2">4 hours ago</p>
                    </div>
                    <div className="p-3 border rounded">
                      <h4 className="font-medium">Fed Signals Rate Pause</h4>
                      <p className="text-sm text-slate-600 mt-1">
                        Federal Reserve hints at maintaining current rates...
                      </p>
                      <p className="text-xs text-slate-500 mt-2">6 hours ago</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
