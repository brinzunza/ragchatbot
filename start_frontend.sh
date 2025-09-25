#!/bin/bash

# Start the React frontend
echo "starting ai assistant frontend..."

cd frontend

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "installing frontend dependencies..."
    npm install
fi

# Start the React development server
echo "starting react development server on http://localhost:3000"
npm start