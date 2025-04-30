import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// List of valid paths
const validPaths = ['/', '/game/[gameId]']

export function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname

  // Check if the path is valid
  const isValidPath = validPaths.some(validPath => {
    if (validPath.includes('[gameId]')) {
      // Handle dynamic paths like /game/[gameId]
      return path.startsWith('/game/') && path.split('/').length === 3
    }
    return path === validPath
  })

  // If the path is not valid, redirect to home
  if (!isValidPath) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  return NextResponse.next()
}

// Configure which paths the middleware should run on
export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
} 