# DeltaQuest API Curl Commands

## Authentication
All commands use your Alpaca API credentials:
- **API Key**: `AKAH5A3VGCR3S9FNWIVC`
- **Secret Key**: `dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy`

## Account Information

### Get Account Details
**Description**: Retrieves your account information including status, options trading level, and buying power.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/account"
```

**Response**: Shows account status, options approval level, buying power, cash balance, etc.

---

## Stock Data

### Get Latest Stock Quote
**Description**: Retrieves real-time bid/ask quotes for a stock symbol.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v2/stocks/AAPL/quotes/latest"
```

**Response**: Returns bid price, ask price, and timestamp for the latest quote.

### Get Latest Stock Trade
**Description**: Retrieves the most recent trade price for a stock symbol.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v2/stocks/AAPL/trades/latest"
```

**Response**: Returns the latest trade price, size, and timestamp.

### Get Recent Stock Bars
**Description**: Retrieves recent daily bars (OHLCV data) for a stock symbol.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Day&start=2025-07-07&end=2025-07-14&limit=5"
```

**Response**: Returns array of daily bars with open, high, low, close, and volume.

---

## Options Data

### Get Options Contracts
**Description**: Retrieves available option contracts for an underlying stock symbol.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/options/contracts?underlying_symbols=AAPL&limit=10"
```

**Response**: Returns array of option contracts with symbol, strike price, expiration date, and contract type.

### Get Options Contracts for Specific Date
**Description**: Retrieves option contracts expiring on a specific date.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/options/contracts?underlying_symbols=TSLA&expiration_date=2025-07-18&limit=100"
```

**Response**: Returns option contracts for the specified expiration date.

### Get Option Quote
**Description**: Retrieves real-time bid/ask quotes for a specific option contract.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v2/options/AAPL240719C00190000/quotes/latest"
```

**Response**: Returns bid price, ask price, and timestamp for the option contract.

---

## Testing Commands

### Test Paper Trading (Will Fail)
**Description**: This will return 401 unauthorized because we're using Live API keys.

```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://paper-api.alpaca.markets/v2/account"
```

**Response**: `{"code":40110000,"message":"request is not authorized"}`

### Verbose Test for Debugging
**Description**: Uses `-v` flag to show detailed request/response information.

```bash
curl -v -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/account"
```

**Response**: Shows full HTTP headers, SSL info, and response details.

---

## Key Discoveries

1. **API Keys**: Our keys start with "AK" = Live Trading keys (not Paper Trading)
2. **Options Enabled**: Account has `options_approved_level: 1` and `options_trading_level: 1`
3. **Working Endpoints**: Live API (`https://api.alpaca.markets`) works, Paper API fails
4. **Stock Data**: Real-time quotes working via `https://data.alpaca.markets`
5. **Options Data**: Options contracts and quotes accessible via Live API

## Environment Variables

```bash
ALPACA_API_KEY=AKAH5A3VGCR3S9FNWIVC
ALPACA_SECRET_KEY=dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy
ALPACA_PAPER_TRADING=false  # Must be false for these keys


# Test account connection
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/account"

# Get JNJ options chain
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/options/contracts?underlying_symbols=JNJ"


     curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols=JNJ250718P00149000"

       curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols=JNJ"


curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/options/contracts?underlying_symbols=AFL&expiration_date=2025-07-18&type=put&strike_price_lte=90"

```