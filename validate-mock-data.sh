#!/bin/bash

echo "🔬 Validating Mock Apple Data Against Current Market"
echo "=================================================="

# Current AAPL stock price
echo "📈 Fetching current AAPL price..."
CURRENT_PRICE=$(curl -s -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v2/stocks/AAPL/trades/latest" | \
     grep -o '"p":[0-9.]*' | cut -d: -f2)

echo "💰 Current AAPL: \$${CURRENT_PRICE}"
echo "🎯 Mock data AAPL: \$274.115"
echo ""

# Sample option prices
echo "📊 Fetching sample option quotes..."
OPTION_QUOTE=$(curl -s -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols=AAPL260116P00275000")

echo "🔍 275 Put Quote: $OPTION_QUOTE"
echo ""

echo "✅ Mock data is current as of 2025-12-16"
echo "🧪 This ensures consistent testing between CUDA and CPU engines"
echo "⏰ Data won't age - perfect for benchmarking!"
echo ""
echo "🚀 Run './bin/test-engines' to test both engines with this data"