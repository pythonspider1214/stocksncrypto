import { type NextRequest, NextResponse } from "next/server"
import jwt from "jsonwebtoken"
import bcrypt from "bcryptjs"
import { z } from "zod"

// Validation schemas
const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
})

const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
  firstName: z.string().min(1),
  lastName: z.string().min(1),
})

// Mock database - replace with actual database
const users = new Map()

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action } = body

    if (action === "login") {
      const { email, password } = loginSchema.parse(body)

      // Check if user exists
      const user = users.get(email)
      if (!user) {
        return NextResponse.json({ error: "Invalid credentials" }, { status: 401 })
      }

      // Verify password
      const isValidPassword = await bcrypt.compare(password, user.passwordHash)
      if (!isValidPassword) {
        return NextResponse.json({ error: "Invalid credentials" }, { status: 401 })
      }

      // Generate JWT token
      const token = jwt.sign(
        {
          userId: user.id,
          email: user.email,
          role: user.role,
        },
        process.env.JWT_SECRET || "fallback-secret",
        { expiresIn: "24h" },
      )

      return NextResponse.json({
        token,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role,
        },
      })
    }

    if (action === "register") {
      const { email, password, firstName, lastName } = registerSchema.parse(body)

      // Check if user already exists
      if (users.has(email)) {
        return NextResponse.json({ error: "User already exists" }, { status: 409 })
      }

      // Hash password
      const passwordHash = await bcrypt.hash(password, 12)

      // Create user
      const user = {
        id: crypto.randomUUID(),
        email,
        passwordHash,
        firstName,
        lastName,
        role: "user",
        createdAt: new Date().toISOString(),
        isActive: true,
        emailVerified: false,
      }

      users.set(email, user)

      // Generate JWT token
      const token = jwt.sign(
        {
          userId: user.id,
          email: user.email,
          role: user.role,
        },
        process.env.JWT_SECRET || "fallback-secret",
        { expiresIn: "24h" },
      )

      return NextResponse.json({
        token,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role,
        },
      })
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 })
  } catch (error) {
    console.error("Auth error:", error)

    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: "Invalid input", details: error.errors }, { status: 400 })
    }

    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get("authorization")

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return NextResponse.json({ error: "No token provided" }, { status: 401 })
    }

    const token = authHeader.substring(7)

    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET || "fallback-secret") as any

      // Get user from database
      const user = Array.from(users.values()).find((u) => u.id === decoded.userId)

      if (!user) {
        return NextResponse.json({ error: "User not found" }, { status: 404 })
      }

      return NextResponse.json({
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role,
        },
      })
    } catch (jwtError) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }
  } catch (error) {
    console.error("Auth verification error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
