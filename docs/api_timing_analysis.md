# Alpaca API Timing Analysis

## Official Rate Limits

Based on official Alpaca Market Data API documentation:

### Basic Plan (Free)
- **Rate Limit**: 200 requests per minute
- **Per Second**: ~3.33 requests per second
- **Recommended Delay**: 350ms between requests (2.86 req/sec for safety)

### Algo Trader Plus ($99/month)
- **Rate Limit**: 10,000 requests per minute  
- **Per Second**: ~166.67 requests per second
- **Recommended Delay**: 10ms between requests (100 req/sec for safety)

## Current Implementation

### Batch Structure
- **5 batches** of **100 stocks** each = 500 stocks total
- **2 options requests per stock** (1 puts + 1 calls)
- **Total API calls**: 5 stock batches + (500 × 2) options = **1,005 requests**

### Timing Strategy (Basic Plan)
```
Stock batch request → No delay needed (1 request)
↓
For each stock in batch:
  └── Puts request → 350ms delay → Calls request → 350ms delay
↓  
Next batch → 1 second delay
```

### Time Calculation (Basic Plan)
```
Per stock: 2 requests × 350ms = 700ms
Per batch: 100 stocks × 700ms = 70 seconds  
Between batches: 1 second × 4 gaps = 4 seconds
Total time: (70 × 5) + 4 = 354 seconds (~6 minutes)
```

### Time Calculation (Algo Trader Plus)
```
Per stock: 2 requests × 10ms = 20ms
Per batch: 100 stocks × 20ms = 2 seconds
Between batches: 100ms × 4 gaps = 400ms  
Total time: (2 × 5) + 0.4 = 10.4 seconds
```

## Performance Comparison

| Plan | Delay per Request | Total Time | Requests/Min Used |
|------|-------------------|------------|------------------|
| Basic (Free) | 350ms | ~6 minutes | ~170/200 (85%) |
| Algo Trader Plus | 10ms | ~10 seconds | ~6,000/10,000 (60%) |

## Current Code Implementation

### main.go Timing
- Between puts/calls: **350ms** (Basic plan safe)
- Between stocks: **350ms** (Basic plan safe) 
- Between batches: **1 second** (buffer for safety)

### Alpaca Client Timing  
- Between option batches: **500ms** (conservative)
- Between stock requests: **300ms** (conservative)

## Recommendations

### For Basic Plan Users
✅ Current timing is optimal and safe
- Stays well under 200 req/min limit
- Provides good error margin
- Completes S&P 500 analysis in ~6 minutes

### For Algo Trader Plus Users
⚡ Could optimize for much faster execution:
- Reduce delays to 10ms between requests
- Complete analysis in ~10 seconds
- Add plan detection logic for automatic optimization

### Future Optimizations
1. **Plan Detection**: Auto-detect user's plan and adjust timing
2. **Adaptive Rate Limiting**: Monitor response headers for rate limit info
3. **Concurrent Batches**: Process multiple stocks in parallel for Algo Trader Plus
4. **Exponential Backoff**: Handle rate limit errors gracefully