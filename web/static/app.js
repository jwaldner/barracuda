// Global variables
let selectedDelta = 0.25; // Default delta value



document.addEventListener('DOMContentLoaded', function() {
    // ✅ THIS JAVASCRIPT CHANGE REQUIRES NO REBUILD!
    console.log('🚀 Web assets loaded - no rebuild needed for web changes!');
    
    // Use Go-calculated expiration date, with JavaScript fallback
    const datePickerEl = document.getElementById('expirationDate');
    if (datePickerEl && (!datePickerEl.value || datePickerEl.value === '')) {
        // Use Go value if available, otherwise calculate in JavaScript
        if (window.DEFAULT_EXPIRATION_DATE) {
            datePickerEl.value = window.DEFAULT_EXPIRATION_DATE;
            console.log('📅 Using Go-calculated expiration date:', window.DEFAULT_EXPIRATION_DATE);
        } else {
            // JavaScript fallback (simple next Friday)
            const today = new Date();
            const dayOfWeek = today.getDay();
            const daysUntilFriday = (5 - dayOfWeek + 7) % 7 || 7;
            const nextFriday = new Date(today);
            nextFriday.setDate(today.getDate() + daysUntilFriday);
            datePickerEl.value = nextFriday.toISOString().split('T')[0];
            console.log('📅 Using JavaScript fallback expiration date');
        }
    }
    
    // Setup risk selector event listeners
    setupRiskSelector();
    
    // Set default risk level from config
    const runButton = document.getElementById('runAnalysis');
    if (runButton) {
        const defaultRisk = runButton.dataset.defaultRisk || 'LOW';
        setDefaultRiskLevel(defaultRisk);
        
        runButton.addEventListener('click', async function(e) {
        e.preventDefault();
        
        const loadingDiv = document.getElementById('loading');
        const resultsDiv = document.getElementById('results');
        
        if (loadingDiv) {
            loadingDiv.classList.remove('hidden');
        }
        if (resultsDiv) {
            resultsDiv.classList.add('hidden');
        }
        
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
                symbols = data.symbols.map(s => typeof s === 'string' ? s : s.symbol);
                console.log('Using ALL S&P 500 symbols:', symbols.length, 'symbols');
            } catch (error) {
                console.error('Error fetching S&P 500 symbols:', error);
                alert('Error loading S&P 500 symbols: ' + error.message);
                return;
            }
        } else {
            // Use configured symbol list (your 8 stocks)
            symbols = symbolsData.split(',').map(s => s.trim()).filter(s => s);
            console.log('Using configured symbols:', symbols);
        }
        // Get available cash from the cash input (only updated when analyze is clicked)
        const cashInputEl = document.getElementById('cashAmount');
        const availableCash = parseInt(cashInputEl ? cashInputEl.value : 0) || 70000;
        console.log('💰 Using cash amount:', availableCash);
        
        const contractsEl = document.getElementById('contractsProcessed');
        if (contractsEl && contractsEl.querySelector('p')) {
            contractsEl.querySelector('p').textContent = symbols.length;
        }
        
        // Update workload status to show processing
        document.getElementById('workloadStatus').innerHTML = '<small>🔥 WORKLOAD: ACTIVE - Processing ' + symbols.length + ' symbols</small>';
        // No styling - use existing Tailwind classes
        
        // Get expiration date from date picker (set by backend)
        const datePickerEl = document.getElementById('expirationDate');
        const selectedDate = datePickerEl ? datePickerEl.value : '';
        
        const analysisData = {
            symbols: symbols,
            expiration_date: selectedDate,
            target_delta: selectedDelta,
            available_cash: availableCash,
            strategy: "puts"
        };
        
        console.log('Sending analysis request with delta:', selectedDelta, 'data:', analysisData);
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(analysisData)
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
                // No styling - use existing Tailwind classes
                
                // Reset to idle after 3 seconds
                setTimeout(() => {
                    const workloadStatusTimeout = document.getElementById('workloadStatus');
                    if (workloadStatusTimeout) {
                        workloadStatusTimeout.innerHTML = '<small>🔥 WORKLOAD: IDLE</small>';
                        // No styling - use existing Tailwind classes
                    }
                }, 3000);
            }
        } catch (error) {
            alert('Error: ' + error.message);
            
            // Update workload status to show error
            const workloadStatus = document.getElementById('workloadStatus');
            if (workloadStatus) {
                workloadStatus.innerHTML = '<small>🔥 WORKLOAD: ERROR</small>';
                // No styling - use existing Tailwind classes
            }
        } finally {
            if (loadingDiv) {
                loadingDiv.classList.add('hidden');
            }
        }
        });
    }
    
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
    const resultsDiv = document.getElementById('results');
    const tbody = document.getElementById('results-body');
    
    if (!tbody) {
        console.error('Results table body not found');
        return;
    }
    
    if (!data.results || data.results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="p-8 text-center text-gray-500">No suitable put options found.</td></tr>';
        if (resultsDiv) {
            resultsDiv.classList.remove('hidden');
        }
        const exportBtn = document.getElementById('exportCSV');
        if (exportBtn) {
            exportBtn.disabled = true;
        }
        return;
    }
    
    const exportBtn = document.getElementById('exportCSV');
    if (exportBtn) {
        exportBtn.disabled = false;
    }
    window.lastResults = data.results;
    if (tbody) {
        tbody.innerHTML = '';
    }
    
    data.results.forEach((option, index) => {
        const rank = index + 1;
        const isFirst = index === 0;
        
        const row = document.createElement('tr');
        row.className = isFirst ? 'bg-green-50 border-l-4 border-green-500 hover:bg-green-100' : 'hover:bg-gray-50';
        
        // Calculate annualized return: (profit_percentage / days) * 365
        const annualizedReturn = option.profit_percentage && option.days_to_expiration 
            ? ((option.profit_percentage / option.days_to_expiration) * 365).toFixed(1)
            : 'N/A';

        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">${rank}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">${option.ticker}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">$${option.strike.toFixed(2)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">$${option.stock_price.toFixed(2)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${option.max_contracts}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">$${option.premium.toFixed(2)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-green-600">$${option.total_premium.toLocaleString()}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">$${(option.cash_needed ? option.cash_needed.toLocaleString() : 'N/A')}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-green-600">${(option.profit_percentage ? option.profit_percentage.toFixed(2) + '%' : 'N/A')}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-blue-600">${annualizedReturn}%</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${(option.expiration || 'N/A')}</td>
        `;
        
        if (tbody) {
            tbody.appendChild(row);
        }
    });
    
    if (resultsDiv) {
        resultsDiv.classList.remove('hidden');
    }
}

function showNotification() {
    const notification = document.getElementById('notification');
    if (notification) {
        notification.classList.remove('hidden');
        setTimeout(() => {
            const notificationTimeout = document.getElementById('notification');
            if (notificationTimeout) {
                notificationTimeout.classList.add('hidden');
            }
        }, 3000);
    }
}

async function copyTableToCSV() {
    const table = document.querySelector('#results table');
    if (!table) {
        alert('No table data available to copy');
        return;
    }

    const rows = table.querySelectorAll('tr');
    const csvContent = [];

    // Process each row
    rows.forEach(row => {
        const cells = row.querySelectorAll('th, td');
        const rowData = [];
        
        cells.forEach(cell => {
            // Get text content and clean it up
            let text = cell.textContent.trim();
            
            // Remove extra whitespace and newlines
            text = text.replace(/\s+/g, ' ');
            
            // Handle commas in values by wrapping in quotes
            if (text.includes(',')) {
                text = `"${text}"`;
            }
            
            rowData.push(text);
        });
        
        csvContent.push(rowData.join(','));
    });

    const csvString = csvContent.join('\n');

    try {
        await navigator.clipboard.writeText(csvString);
        
        // Visual feedback
        const btn = document.getElementById('copy-csv-btn');
        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '✅ Copied!';
            btn.classList.add('bg-green-500/40');
            
            setTimeout(() => {
                const btnTimeout = document.getElementById('copy-csv-btn');
                if (btnTimeout) {
                    btnTimeout.innerHTML = originalText;
                    btnTimeout.classList.remove('bg-green-500/40');
                }
            }, 2000);
        }
        
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = csvString;
        document.body.appendChild(textArea);
        textArea.select();
        
        try {
            document.execCommand('copy');
            alert('CSV data copied to clipboard');
        } catch (fallbackErr) {
            alert('Failed to copy to clipboard. Please copy manually:\\n\\n' + csvString);
        }
        
        document.body.removeChild(textArea);
    }
}

// Risk selector functions
function setupRiskSelector() {
    document.querySelectorAll('.risk-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const delta = parseFloat(btn.dataset.delta);
            console.log('Risk button clicked, delta:', delta);
            selectRisk(delta);
            console.log('selectedDelta updated to:', selectedDelta);
            
            // Auto-trigger analysis if there are symbols
            const runButton = document.getElementById('runAnalysis');
            const symbols = runButton.dataset.symbols;
            if (symbols && symbols.trim()) {
                console.log('Auto-triggering analysis with delta:', selectedDelta);
                setTimeout(() => {
                    document.getElementById('runAnalysis').click();
                }, 300);
            }
        });
    });
}

function selectRisk(delta) {
    selectedDelta = delta;

    // Reset ALL buttons to default state first
    document.querySelectorAll('.risk-btn').forEach(btn => {
        const btnDelta = parseFloat(btn.dataset.delta);

        // Reset to base classes
        btn.className = 'risk-btn px-4 py-3 rounded-xl text-sm font-bold text-white transition-all transform duration-300 hover:scale-105';

        // Add appropriate color classes based on delta
        if (btnDelta === 0.25) {
            btn.classList.add('bg-green-600/40', 'border-2', 'border-green-400', 'hover:bg-green-600/60');
        } else if (btnDelta === 0.50) {
            btn.classList.add('bg-orange-600/40', 'border-2', 'border-orange-400', 'hover:bg-orange-600/60');
        } else if (btnDelta === 0.75) {
            btn.classList.add('bg-red-600/40', 'border-2', 'border-red-400', 'hover:bg-red-600/60');
        }

        // Apply selected styling to the matching button
        if (Math.abs(btnDelta - delta) < 0.001) {
            btn.className = 'risk-btn px-4 py-3 rounded-xl text-sm font-bold transition-all transform duration-300 bg-white text-black border-4 border-white shadow-2xl scale-110 ring-4 ring-white/50';
        }
    });
}

function setDefaultRiskLevel(riskLevel) {
    const deltaMap = {
        'LOW': 0.25,
        'MOD': 0.50,
        'HIGH': 0.75
    };
    
    const delta = deltaMap[riskLevel] || 0.25;
    selectRisk(delta);
}

// DeltaQuest-specific functions
// Global strategy state - puts only
let currentStrategy = 'puts';

function switchStrategy(strategy) {
    currentStrategy = strategy;

    // Update tab styling
    document.querySelectorAll('.strategy-tab').forEach(tab => {
        tab.classList.remove('bg-blue-600', 'text-white');
        tab.classList.add('text-slate-300');
    });

    const activeTab = strategy === 'puts' ? 'puts-tab' : 'calls-tab';
    const activeElement = document.getElementById(activeTab);
    if (activeElement) {
        activeElement.classList.remove('text-slate-300');
        activeElement.classList.add('bg-blue-600', 'text-white');
    }
}

function selectRiskDeltaQuest(delta) {
    selectedDelta = delta;

    // Reset ALL buttons to default state first
    document.querySelectorAll('.risk-btn').forEach(btn => {
        const btnDelta = parseFloat(btn.dataset.delta);

        // Reset to base classes and add appropriate gradient
        if (btnDelta === 0.25) {
            btn.className = 'risk-btn px-6 py-4 rounded-lg text-sm font-bold text-white bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200';
        } else if (btnDelta === 0.50) {
            btn.className = 'risk-btn px-6 py-4 rounded-lg text-sm font-bold text-white bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-500 hover:to-orange-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200';
        } else if (btnDelta === 0.75) {
            btn.className = 'risk-btn px-6 py-4 rounded-lg text-sm font-bold text-white bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200';
        }

        // Apply selected styling to the matching button
        if (Math.abs(btnDelta - delta) < 0.001) {
            btn.classList.add('ring-4', 'ring-blue-400', 'ring-opacity-75', 'scale-105');
        }
    });
}

async function handleSubmit(e) {
    e.preventDefault();

    // Validate selectedDelta before proceeding
    if (isNaN(selectedDelta) || selectedDelta <= 0) {
        selectedDelta = 0.50;
    }

    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    const errorEl = document.getElementById('error-message');
    if (errorEl) errorEl.classList.add('hidden');

    // Get form values
    const symbolsValue = document.getElementById('symbols').value;
    const expirationValue = document.getElementById('expiration-date').value;
    const cashValue = document.getElementById('available-cash').value;

    // Validate required fields
    if (!symbolsValue.trim()) {
        showError('Please enter at least one stock symbol');
        document.getElementById('loading').classList.add('hidden');
        return;
    }

    if (!expirationValue) {
        showError('Please select an expiration date');
        document.getElementById('loading').classList.add('hidden');
        return;
    }

    // Create form data
    const params = new URLSearchParams();
    params.append('symbols', symbolsValue);
    params.append('expiration_date', expirationValue);
    params.append('target_delta', selectedDelta.toString());
    params.append('available_cash', cashValue);
    params.append('strategy', currentStrategy);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: params.toString()
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error('HTTP error! status: ' + response.status + ', message: ' + errorText);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError('Analysis failed: ' + error.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
}



function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
        setTimeout(function() { errorDiv.classList.add('hidden'); }, 8000);
    }
}

async function copyTableToCSV() {
    const table = document.querySelector('#results table');
    if (!table) {
        alert('No table data available to copy');
        return;
    }

    const rows = table.querySelectorAll('tr');
    const csvContent = [];

    // Process each row
    rows.forEach(function(row) {
        const cells = row.querySelectorAll('th, td');
        const rowData = [];
        
        cells.forEach(function(cell) {
            // Get text content and clean it up
            let text = cell.textContent.trim();
            
            // Remove extra whitespace and newlines
            text = text.replace(/\s+/g, ' ');
            
            // Handle commas in values by wrapping in quotes
            if (text.includes(',')) {
                text = '"' + text + '"';
            }
            
            rowData.push(text);
        });
        
        csvContent.push(rowData.join(','));
    });

    const csvString = csvContent.join('\n');

    try {
        await navigator.clipboard.writeText(csvString);
        
        // Visual feedback
        const btn = document.getElementById('copy-csv-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '✅ Copied!';
        btn.classList.add('bg-green-500/40');
        
        setTimeout(function() {
            btn.innerHTML = originalText;
            btn.classList.remove('bg-green-500/40');
        }, 2000);
        
    } catch (err) {
        alert('Failed to copy to clipboard');
    }
}

// Initialize DeltaQuest interface
document.addEventListener('DOMContentLoaded', function () {
    // Puts-only strategy - no tab switching needed

    // Setup risk buttons with DeltaQuest styling
    document.querySelectorAll('.risk-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const delta = parseFloat(btn.dataset.delta);
            selectRiskDeltaQuest(delta);
        });
    });

    // Setup form submission
    const form = document.getElementById('analysis-form');
    if (form) form.addEventListener('submit', handleSubmit);
    
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) analyzeBtn.addEventListener('click', handleSubmit);

    // Set default strategy and risk level
    switchStrategy('puts');
    setTimeout(() => {
        selectRiskDeltaQuest(0.50);
    }, 100);
});