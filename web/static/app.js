// Calculate 3rd Friday of the month
function getThirdFriday() {
    const today = new Date();
    const currentMonth = today.getMonth();
    const currentYear = today.getFullYear();
    
    // Find 3rd Friday of current month
    const firstDay = new Date(currentYear, currentMonth, 1);
    const firstFriday = new Date(firstDay);
    firstFriday.setDate(1 + (5 - firstDay.getDay() + 7) % 7);
    const thirdFriday = new Date(firstFriday);
    thirdFriday.setDate(firstFriday.getDate() + 14);
    
    // If current day is in the week of 3rd Friday or past it, use next month's 3rd Friday
    const weekStart = new Date(thirdFriday);
    weekStart.setDate(thirdFriday.getDate() - 7);
    
    if (today >= weekStart) {
        // Use next month's 3rd Friday
        const nextMonth = currentMonth === 11 ? 0 : currentMonth + 1;
        const nextYear = currentMonth === 11 ? currentYear + 1 : currentYear;
        const nextFirstDay = new Date(nextYear, nextMonth, 1);
        const nextFirstFriday = new Date(nextFirstDay);
        nextFirstFriday.setDate(1 + (5 - nextFirstDay.getDay() + 7) % 7);
        const nextThirdFriday = new Date(nextFirstFriday);
        nextThirdFriday.setDate(nextFirstFriday.getDate() + 14);
        return nextThirdFriday;
    }
    
    return new Date(year, month, day);
}



document.addEventListener('DOMContentLoaded', function() {
    const thirdFriday = getThirdFriday();
    document.getElementById('autoExpiration').textContent = thirdFriday.toLocaleDateString();
    
    document.getElementById('runAnalysis').addEventListener('click', async function(e) {
        e.preventDefault();
        
        const loadingDiv = document.getElementById('loading');
        const resultsDiv = document.getElementById('results');
        
        loadingDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
        
        // Get symbols - either from config or S&P 500 assets
        let symbols;
        const symbolsData = e.target.dataset.symbols;
        const useSP500 = e.target.dataset.useSp500 === 'true';
        
        if (useSP500 || !symbolsData || !symbolsData.trim()) {
            // Empty config list - get ALL S&P 500 symbols from assets
            try {
                const response = await fetch('/api/sp500/symbols');
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                const data = await response.json();
                if (!data.symbols || data.symbols.length === 0) {
                    throw new Error('No S&P 500 symbols available');
                }
                // Extract just the symbol strings from the full symbol objects
                symbols = data.symbols.map(s => typeof s === 'string' ? s : s.symbol);
                console.log('Using ALL S&P 500 symbols from assets:', symbols.length, 'symbols (will process in batches, show top 25 results)');
            } catch (error) {
                console.error('Error fetching S&P 500 symbols:', error);
                alert('Error loading S&P 500 symbols: ' + error.message);
                return;
            }
        } else {
            // Use configured symbol list
            symbols = symbolsData.split(',').map(s => s.trim()).filter(s => s);
            console.log('Using configured symbols:', symbols);
        }
        const defaultCash = e.target.dataset.defaultCash ? 
            parseInt(e.target.dataset.defaultCash) : 
            70000;
        
        document.getElementById('contractsProcessed').querySelector('p').textContent = symbols.length;
        
        // Update workload status to show processing
        const workloadStatus = document.getElementById('workloadStatus');
        if (workloadStatus) {
            workloadStatus.innerHTML = '<small>🔥 WORKLOAD: ACTIVE - Processing ' + symbols.length + ' symbols</small>';
            workloadStatus.style.background = '#ff4444';
            workloadStatus.style.color = '#ffffff';
        }
        
        const thirdFriday = getThirdFriday();
        const data = {
            symbols: symbols,
            expiration_date: thirdFriday.toISOString().split('T')[0],
            target_delta: 0.25,
            available_cash: defaultCash,
            strategy: 'puts'
        };
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Analysis failed');
            }
            
            displayResults(result);
            
            // Update workload status to show completion
            const workloadStatus = document.getElementById('workloadStatus');
            if (workloadStatus && result.processing_time) {
                workloadStatus.innerHTML = '<small>🔥 WORKLOAD: COMPLETED (' + result.processing_time.toFixed(2) + 's) - IDLE</small>';
                workloadStatus.style.background = '#00aa00';
                workloadStatus.style.color = '#ffffff';
                
                // Reset to idle after 3 seconds
                setTimeout(() => {
                    workloadStatus.innerHTML = '<small>🔥 WORKLOAD: IDLE</small>';
                    workloadStatus.style.background = '#333';
                    workloadStatus.style.color = '#00ff00';
                }, 3000);
            }
        } catch (error) {
            alert('Error: ' + error.message);
            
            // Update workload status to show error
            const workloadStatus = document.getElementById('workloadStatus');
            if (workloadStatus) {
                workloadStatus.innerHTML = '<small>🔥 WORKLOAD: ERROR</small>';
                workloadStatus.style.background = '#aa0000';
                workloadStatus.style.color = '#ffffff';
            }
        } finally {
            loadingDiv.style.display = 'none';
        }
    });
    
    // CSV Export functionality
    document.getElementById('exportCSV').addEventListener('click', function() {
        if (!window.lastResults) return;
        
        const headers = ['Rank', 'Symbol', 'Strike', 'Stock_Price', 'Premium_Per_Contract', 'Max_Contracts', 'Total_Premium', 'Expiration'];
        let csvContent = headers.join(',') + '\n';
        
        window.lastResults.forEach((option, index) => {
            const rank = index + 1;
            const row = [
                rank,
                option.ticker,
                option.strike.toFixed(2),
                option.stock_price.toFixed(2),
                option.premium.toFixed(2),
                option.max_contracts,
                option.total_premium.toFixed(2),
                option.expiration || 'N/A'
            ];
            csvContent += row.join(',') + '\n';
        });
        
        // Copy to clipboard
        navigator.clipboard.writeText(csvContent).then(function() {
            showNotification();
        });
    });
});

function displayResults(data) {
    const resultsContent = document.getElementById('resultsContent');
    const resultsDiv = document.getElementById('results');
    
    if (!data.results || data.results.length === 0) {
        resultsContent.innerHTML = '<p>No suitable put options found for the specified criteria.</p>';
        resultsDiv.style.display = 'block';
        document.getElementById('exportCSV').disabled = true;
        return;
    }
    
    // Enable CSV export and store data
    document.getElementById('exportCSV').disabled = false;
    window.lastResults = data.results;
    
    let html = '<div class="table-container">';
    html += '<table class="results-table">';
    html += '<thead>';
    html += '<tr>';
    html += '<th>Rank</th>';
    html += '<th>Symbol</th>';
    html += '<th>Strike</th>';
    html += '<th>Stock Price</th>';
    html += '<th>Contracts</th>';
    html += '<th>Premium/Contract</th>';
    html += '<th>Total Premium</th>';
    html += '<th>Cash Needed</th>';
    html += '<th>Profit %</th>';
    html += '<th>Exp Date</th>';
    html += '</tr>';
    html += '</thead>';
    html += '<tbody>';
    
    data.results.forEach((option, index) => {
        const rank = index + 1;
        html += '<tr>';
        html += '<td>' + rank + '</td>';
        html += '<td>' + option.ticker + '</td>';
        html += '<td>' + option.strike.toFixed(2) + '</td>';
        html += '<td>' + option.stock_price.toFixed(2) + '</td>';
        html += '<td>' + option.max_contracts + '</td>';
        html += '<td>' + option.premium.toFixed(2) + '</td>';
        html += '<td>' + option.total_premium.toFixed(2) + '</td>';
        html += '<td>' + (option.cash_needed ? option.cash_needed.toFixed(2) : 'N/A') + '</td>';
        html += '<td>' + (option.profit_percentage ? option.profit_percentage.toFixed(2) + '%' : 'N/A') + '</td>';
        html += '<td>' + (option.expiration || 'N/A') + '</td>';
        html += '</tr>';
    });
    
    html += '</tbody>';
    html += '</table>';
    html += '</div>';
    
    // Add processing statistics if available
    if (data.processing_stats) {
        html += '<div class="performance-footer">';
        html += '<strong>Performance:</strong> ';
        
        // Highlight CUDA/CPU status with bright yellow background
        let stats = data.processing_stats;
        
        // Handle different CUDA status patterns with highly visible yellow highlighting
        if (stats.includes('CUDA: true')) {
            stats = stats.replace(/CUDA: true/, `<span style="background: #FFFF00; color: #000000; padding: 4px 8px; border-radius: 4px; font-weight: 900; font-size: 12px; text-shadow: 1px 1px 0px #888; border: 2px solid #000; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">CUDA</span>`);
        } else if (stats.includes('CUDA: false')) {
            stats = stats.replace(/CUDA: false/, `<span style="background: #FFFF00; color: #000000; padding: 4px 8px; border-radius: 4px; font-weight: 900; font-size: 12px; text-shadow: 1px 1px 0px #888; border: 2px solid #000; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">CPU</span>`);
        } else if (stats.toLowerCase().includes('cuda')) {
            // Fallback: highlight any occurrence of the word cuda
            stats = stats.replace(/cuda/gi, `<span style="background: #FFFF00; color: #000000; padding: 4px 8px; border-radius: 4px; font-weight: 900; font-size: 12px; text-shadow: 1px 1px 0px #888; border: 2px solid #000; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">CUDA</span>`);
        }
        
        html += stats;
        html += '</div>';
    }
    
    resultsContent.innerHTML = html;
    resultsDiv.style.display = 'block';
}

function showNotification() {
    const notification = document.getElementById('notification');
    notification.style.display = 'block';
    setTimeout(() => {
        notification.style.display = 'none';
    }, 3000);
}