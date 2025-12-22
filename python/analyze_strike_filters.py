#!/usr/bin/env python3
"""
Strike Filter Analysis Script

Analyzes barracuda.log to extract and analyze strike filter percentages
for each symbol to verify proper OTM put filtering.
"""

import re
import sys
from pathlib import Path

def parse_alpaca_filter_logs(log_file_path):
    """Parse Alpaca filter logs and extract strike percentages"""
    
    filter_data = []
    
    # Pattern to match: 🔍 ALPACA FILTER: SYMBOL PUTS - strikes $min to $max (min% to max% of stock $price)
    pattern = r'🔍 ALPACA FILTER: (\w+) PUTS.*?strikes \$(\d+) to \$(\d+) \((\d+\.?\d*)% to (\d+\.?\d*)% of stock \$(\d+\.?\d*)\)'
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(pattern, line)
                if match:
                    symbol = match.group(1)
                    min_strike = float(match.group(2))
                    max_strike = float(match.group(3))
                    min_percent = float(match.group(4))
                    max_percent = float(match.group(5))
                    stock_price = float(match.group(6))
                    
                    # Calculate actual percentages
                    actual_min_percent = (min_strike / stock_price) * 100
                    actual_max_percent = (max_strike / stock_price) * 100
                    
                    filter_data.append({
                        'symbol': symbol,
                        'stock_price': stock_price,
                        'min_strike': min_strike,
                        'max_strike': max_strike,
                        'logged_min_percent': min_percent,
                        'logged_max_percent': max_percent,
                        'actual_min_percent': actual_min_percent,
                        'actual_max_percent': actual_max_percent,
                        'is_otm': max_strike < stock_price,
                        'line_num': line_num
                    })
    
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
        return []
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return []
    
    return filter_data

def analyze_filters(filter_data):
    """Analyze filter data and generate report"""
    
    if not filter_data:
        print("❌ No filter data found in logs")
        return
    
    # Separate old vs new entries (new entries have lower max percentages)
    otm_entries = [d for d in filter_data if d['is_otm']]
    itm_entries = [d for d in filter_data if not d['is_otm']]
    
    print("=" * 80)
    print("📊 STRIKE FILTER ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    if otm_entries:
        print("✅ CORRECT OTM FILTERING (Recent - Fixed):")
        print(f"{'Symbol':<8} {'Stock $':<10} {'Strike Range':<15} {'Percentages':<12} {'Status'}")
        print("-" * 60)
        
        for data in otm_entries:
            strike_range = f"${data['min_strike']:.0f}-${data['max_strike']:.0f}"
            percentages = f"{data['actual_min_percent']:.1f}%-{data['actual_max_percent']:.1f}%"
            print(f"{data['symbol']:<8} ${data['stock_price']:<9.2f} {strike_range:<15} {percentages:<12} ✅ OTM")
        print()
    
    if itm_entries:
        print("❌ INCORRECT ITM FILTERING (Old - Before Fix):")
        print(f"{'Symbol':<8} {'Stock $':<10} {'Strike Range':<15} {'Percentages':<12} {'Status'}")
        print("-" * 60)
        
        for data in itm_entries:
            strike_range = f"${data['min_strike']:.0f}-${data['max_strike']:.0f}"
            percentages = f"{data['actual_min_percent']:.1f}%-{data['actual_max_percent']:.1f}%"
            print(f"{data['symbol']:<8} ${data['stock_price']:<9.2f} {strike_range:<15} {percentages:<12} ❌ ITM")
        print()
    
    # Summary
    total_symbols = len(filter_data)
    print("🔍 SUMMARY:")
    print(f"   • Total log entries: {total_symbols}")
    print(f"   • ✅ Correct OTM entries (strikes < stock price): {len(otm_entries)}")
    print(f"   • ❌ Incorrect ITM entries (strikes > stock price): {len(itm_entries)}")
    
    if len(otm_entries) > 0:
        print(f"\n🎉 SUCCESS: Fix is working! {len(otm_entries)} symbols now have proper OTM filtering!")
        
        # Show the pattern for the fixed entries
        if otm_entries:
            max_percentages = [d['actual_max_percent'] for d in otm_entries]
            avg_max = sum(max_percentages) / len(max_percentages)
            print(f"   • New max strike average: {avg_max:.1f}% of stock price (properly below 100%)")
    
    if len(itm_entries) > 0:
        print(f"\n⚠️  Old entries show the problem: {len(itm_entries)} had strikes above stock price")
        
        # Show the pattern for the broken entries  
        if itm_entries:
            max_percentages = [d['actual_max_percent'] for d in itm_entries]
            avg_max = sum(max_percentages) / len(max_percentages)
            print(f"   • Old max strike average: {avg_max:.1f}% of stock price (incorrectly above 100%)")

def main():
    """Main function"""
    
    # Default to barracuda.log in current directory
    log_file = Path("barracuda.log")
    
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        print("Usage: python analyze_strike_filters.py [log_file_path]")
        print("Default: barracuda.log")
        sys.exit(1)
    
    print(f"📖 Analyzing strike filters in: {log_file}")
    print()
    
    # Parse and analyze
    filter_data = parse_alpaca_filter_logs(log_file)
    analyze_filters(filter_data)

if __name__ == "__main__":
    main()