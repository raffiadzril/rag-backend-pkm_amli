#!/bin/bash

echo "Testing MPASI Menu Plan API..."
echo "================================"
echo ""

curl -s -X 'GET' \
  'http://127.0.0.1:8000/api/menu-plan?age_months=8&weight_kg=9&height_cm=72&residence=Kuningan' \
  -H 'accept: application/json' | python3 -m json.tool

echo ""
echo "================================"
echo "Test complete!"
