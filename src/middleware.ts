import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  console.log("=== INCOMING REQUEST ===");
  console.log("Method:", request.method);
  console.log("URL:", request.url);
  console.log("Path:", new URL(request.url).pathname);
  console.log("Headers:", Object.fromEntries(request.headers));
  console.log("=======================");
  
  return NextResponse.next();
}

// Configure which paths the middleware runs on
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
} 