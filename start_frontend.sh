#!/bin/bash
cd frontend

# Clear Next.js cache
rm -rf .next

# Install dependencies
npm install

# Set environment variables
export NEXT_PUBLIC_API_URL="http://localhost:8001"

# Start development server
npm run dev