// Global variables
let selectedDelta;
let backendConfig;

// Load backend configuration from JSON script tag
function loadBackendConfig() {
    const configElement = document.getElementById('backend-config');
    if (configElement) {
        try {
            backendConfig = JSON.parse(configElement.textContent);
            
            // Set global variables from backend config
            const riskLevel = backendConfig.defaultRiskLevel || 'LOW';
            selectedDelta = (riskLevel === 'LOW') ? 0.10 : 
                          (riskLevel === 'MOD') ? 0.20 : 0.30;
            
            // Make backend data available globally
            window.assetData = backendConfig.assetSymbols || {};
            window.csvHeaders = backendConfig.csvHeaders || [];
            window.errorMessages = backendConfig.errorMessages || {};
            
            console.log('🚀 Barracuda initialized with backend defaults');
            console.log('📊 Default Cash:', backendConfig.defaultCash);
            console.log('📅 Default Expiration:', backendConfig.defaultExpirationDate);
            console.log('🎯 Default Risk Level:', backendConfig.defaultRiskLevel);
            console.log('⚖️ Default Delta:', selectedDelta);
            console.log('📈 Default Stocks:', backendConfig.defaultStocks);
            console.log('🏢 Loaded asset data for', Object.keys(window.assetData).length, 'symbols');
            
            return true;
        } catch (error) {
            console.error('❌ Failed to load backend config:', error);
            return false;
        }
    }
    console.warn('⚠️ Backend config not found');
    return false;
}

document.addEventListener('DOMContentLoaded', function() {
    // Load backend configuration first
    loadBackendConfig();
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
    
    // Set default risk level from backend config
    if (backendConfig && backendConfig.defaultRiskLevel) {
        setDefaultRiskLevel(backendConfig.defaultRiskLevel);
    }
    
    // Mode indicator is handled by backend template functions only
    
    // CSV Copy functionality (Copy to clipboard)  
    const copyCSVBtn = document.getElementById('copy-csv-btn');
    if (copyCSVBtn) {
        copyCSVBtn.addEventListener('click', function() {
            if (!window.lastResults) return;
            
            const headers = window.csvHeaders || [];
            let csvContent = headers.join(',') + '\n';
            
            // Helper function to get raw value for CSV
            const getRawValue = (field) => {
                if (!field) return '';
                if (typeof field === 'object' && field.raw !== undefined) {
                    return field.raw;
                }
                return field;
            };
            
            window.lastResults.forEach((option, index) => {
                const ticker = getRawValue(option.ticker);
                const company = getRawValue(option.company) || ticker;
                const sector = getRawValue(option.sector) || 'Unknown';
                    
                const row = [
                    getRawValue(option.rank) || (index + 1),
                    ticker,
                    company,
                    sector,
                    getRawValue(option.strike),
                    getRawValue(option.stock_price),
                    getRawValue(option.premium),
                    getRawValue(option.max_contracts),
                    getRawValue(option.total_premium),
                    getRawValue(option.profit_percentage),
                    getRawValue(option.annualized),
                    getRawValue(option.expiration)
                ];
                csvContent += row.join(',') + '\n';
            });
            
            // Copy to clipboard
            navigator.clipboard.writeText(csvContent).then(function() {
                showNotification('📋 CSV copied to clipboard!', 'success');
            });
        });
    }
    
    // CSV Download functionality (Download file)
    const downloadCSVBtn = document.getElementById('downloadCSVBtn');
    if (downloadCSVBtn) {
        downloadCSVBtn.addEventListener('click', async function() {
            if (!window.lastAnalysisRequest) {
                showNotification('❌ No analysis data available for download', 'error');
                return;
            }
            
            try {
                // Use current frontend selection for filename
                const csvRequest = Object.assign({}, window.lastAnalysisRequest);
                csvRequest.target_delta = selectedDelta; // Use what's actually selected now
                
                console.log('Making fetch request to /api/download-csv');
                const response = await fetch('/api/download-csv', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(csvRequest)
                });
                
                console.log('Response status:', response.status);
                console.log('Response headers:', response.headers);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.log('Error response:', errorText);
                    throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
                }
                
                // Get filename from response headers or generate default
                const contentDisposition = response.headers.get('Content-Disposition');
                console.log('Content-Disposition:', contentDisposition);
                let filename = 'options-analysis.csv';
                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                    if (filenameMatch) {
                        filename = filenameMatch[1];
                    }
                }
                console.log('Filename:', filename);
                
                // Save the file first
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                
                // Then open the same content in new tab
                setTimeout(() => {
                    window.open(url, '_blank');
                }, 100);
                
                // Clean up
                setTimeout(() => {
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                }, 1000);
                
                // Show notification
                showNotification('💾 CSV saved & opened!', 'success');
                
                // Make notification open the CSV file 
                setTimeout(() => {
                    const notification = document.getElementById('notification');
                    if (notification) {
                        notification.onclick = function() {
                            if (window.lastCSVBlob) {
                                const csvUrl = URL.createObjectURL(window.lastCSVBlob);
                                window.open(csvUrl, '_blank');
                                setTimeout(() => URL.revokeObjectURL(csvUrl), 1000);
                            }
                        };
                    }
                }, 100);
                
            } catch (error) {
                console.error('CSV download failed:', error);
                showNotification(`❌ Download failed: ${error.message}`, 'error');
            }
        });
    }
});

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const tbody = document.getElementById('results-body');
    
    if (!tbody) {
        console.error('Results table body not found');
        return;
    }
    
    // Handle new API response structure 
    const results = data.data ? data.data.results : data.results;
    
    if (!results || results.length === 0) {
        tbody.innerHTML = `<tr><td colspan="13" class="p-8 text-center text-gray-500">${window.errorMessages?.noResults || 'No suitable put options found.'}</td></tr>`;
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
    window.lastResults = results;
    if (tbody) {
        tbody.innerHTML = '';
    }
    
    results.forEach((option, index) => {
        const isFirst = index === 0;
        
        const row = document.createElement('tr');
        row.className = isFirst ? 'bg-green-50 border-l-4 border-green-500 hover:bg-green-100' : 'hover:bg-gray-50';
        
        // Helper function to get value from dual format (display for showing, raw for calculations)
        const getValue = (field, useRaw = false) => {
            if (!field) return 'N/A';
            if (typeof field === 'object' && field.display !== undefined) {
                return useRaw ? field.raw : field.display;
            }
            return field; // Fallback for old format
        };

        // Get CSS class based on field type
        const getCSSClass = (field, baseClass = '') => {
            if (typeof field === 'object' && field.type) {
                switch(field.type) {
                    case 'currency': return baseClass + ' text-right font-mono text-green-600 tabular-nums';
                    case 'percentage': return baseClass + ' text-right font-semibold text-blue-600 tabular-nums';
                    case 'integer': return baseClass + ' text-right font-mono tabular-nums';
                    default: return baseClass + ' text-left';
                }
            }
            return baseClass;
        };

        // Get company and sector from backend API response (NEW: separate fields)
        const ticker = getValue(option.ticker, true);
        const companyName = getValue(option.company) || ticker; // Fallback to ticker
        
        // Fix first item tooltip bug - use asset data lookup only
        const assetInfo = window.assetData && window.assetData[ticker];
        const sectorName = assetInfo ? assetInfo.sector : '';
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">${getValue(option.rank)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">${getValue(option.ticker)}</td>
            <td class="company-tooltip px-6 py-4 whitespace-nowrap text-sm text-gray-700" data-sector="${sectorName}">${companyName}</td>
            <td class="sector-column px-6 py-4 whitespace-nowrap text-sm text-gray-600">${sectorName}</td>
            <td class="${getCSSClass(option.strike, 'px-6 py-4 whitespace-nowrap text-sm')}">${getValue(option.strike)}</td>
            <td class="${getCSSClass(option.stock_price, 'px-6 py-4 whitespace-nowrap text-sm')}">${getValue(option.stock_price)}</td>
            <td class="${getCSSClass(option.max_contracts, 'px-6 py-4 whitespace-nowrap text-sm')}">${getValue(option.max_contracts)}</td>
            <td class="${getCSSClass(option.premium, 'px-6 py-4 whitespace-nowrap text-sm')}">${getValue(option.premium)}</td>
            <td class="${getCSSClass(option.total_premium, 'px-6 py-4 whitespace-nowrap text-sm font-bold')}">${getValue(option.total_premium)}</td>

            <td class="${getCSSClass(option.profit_percentage, 'px-6 py-4 whitespace-nowrap text-sm font-bold')}">${getValue(option.profit_percentage)}</td>
            <td class="${getCSSClass(option.annualized, 'px-6 py-4 whitespace-nowrap text-sm font-bold')}">${getValue(option.annualized)}</td>
            <td class="${getCSSClass(option.expiration, 'px-6 py-4 whitespace-nowrap text-sm text-gray-500')}">${getValue(option.expiration)}</td>
        `;
        
        if (tbody) {
            tbody.appendChild(row);
        }
    });
    
    if (resultsDiv) {
        resultsDiv.classList.remove('hidden');
    }
    
    // Update footer with completion stats (reappear after analysis!)
    const workloadStatus = document.getElementById('workloadStatus');
    if (workloadStatus && data.meta && data.meta.processing_time) {
        const processingTime = data.meta.processing_time.toFixed(3);
        const resultCount = results ? results.length : 0;
        const executionMode = data.meta.execution_mode ? data.meta.execution_mode.toUpperCase() : 'UNKNOWN';
        const workloadFactor = data.meta.workload_factor || 0.0;
        const samplesProcessed = data.meta.samples_processed || 0;
        
        // Use actual records processed from backend
        const contractsProcessed = data.meta.contracts_processed || 0;
        
        // Start with actual contracts processed, ADD workload samples if present
        let totalRecords = contractsProcessed; // Actual option contracts processed by CUDA/CPU
        if (workloadFactor > 0.0 && samplesProcessed > 0) {
            totalRecords += samplesProcessed; // ADD Monte Carlo workload samples
        }
        
        // Show processing time, computation time, and records processed with prominent duration  
        const computeTime = data.meta.compute_duration ? data.meta.compute_duration.toFixed(3) : '0.000';
        let footerContent = `🔥 ${executionMode} | ⏰ ${processingTime}s TOTAL | 🔋 ${computeTime}s ${executionMode} COMPUTE 🔋 | 📊 ${totalRecords.toLocaleString()} RECORDS`;
        
        // ADD workload benchmark info when present
        if (workloadFactor > 0.0 && samplesProcessed > 0) {
            const samples = (samplesProcessed / 1000000).toFixed(1);
            footerContent += ` | 🎯 ${samples}M SAMPLES`;
        }
        
        workloadStatus.innerHTML = footerContent;
        
        // Set workload status color based on execution mode
        if (executionMode === 'CUDA') {
            workloadStatus.className = 'text-lg font-bold text-black bg-yellow-400 px-4 py-1 rounded-full';
        } else {
            workloadStatus.className = 'text-lg font-bold text-white bg-green-600 px-4 py-1 rounded-full';
        }
        
        // Update BOTH header and footer mode indicators to keep them synced
        updateAllModeIndicators(executionMode);
        
        // Make footer visible again
        workloadStatus.style.display = 'inline';
        
        // Restore footer when job completes
        const statusFooter = document.getElementById('statusFooter');
        if (statusFooter) {
            statusFooter.style.display = 'block';
        }
    }
}

function showNotification(message = 'CSV data copied to clipboard!', type = 'success', persistent = false) {
	const notification = document.getElementById('notification');
	if (notification) {
		// Update notification message 
		const messageEl = document.getElementById('notification-message');
		if (messageEl) {
			messageEl.textContent = message;
		}
		
		// Update notification style based on type - add cursor pointer for clickability
		if (type === 'error') {
			notification.className = 'fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm bg-red-500 text-white cursor-pointer';
			// Only add default click handler for errors
			notification.onclick = function() {
				hideNotification();
			};
		} else {
			notification.className = 'fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm bg-green-500 text-white cursor-pointer';
			// Don't override onclick for success - let caller set it
		}
		
		notification.classList.remove('hidden');
		
		// Only auto-hide if not persistent
		if (!persistent) {
			setTimeout(() => {
				hideNotification();
			}, 3000);
		}
	}
}

function hideNotification() {
	const notification = document.getElementById('notification');
	if (notification) {
		notification.classList.add('hidden');
	}
}

function showStickyDownloadNotification() {
	const notification = document.getElementById('notification');
	if (notification) {
		const messageEl = document.getElementById('notification-message');
		if (messageEl) {
			// Generic download notification - NO individual file names
			messageEl.innerHTML = `💾 CSV downloaded! <a href="chrome://downloads/" target="_blank" style="color: #90EE90; text-decoration: underline; font-weight: bold;">Open Downloads Folder</a>`;
		}
		
		// Green sticky notification
		notification.className = 'fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm bg-green-500 text-white';
		
		// Remove the general click handler since we have a specific link
		notification.onclick = null;
		
		notification.classList.remove('hidden');
		// Sticky - no auto-hide
	}
}

function openDownloadsFolder() {
	// Just show the keyboard shortcut - no navigation
	const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
	const shortcut = isMac ? 'Cmd+Shift+J' : 'Ctrl+Shift+J';
	showNotification(`📁 CSV saved! Press ${shortcut} to view Downloads`, 'success', true);
}async function copyTableToCSV() {
    const table = document.querySelector('#results table');
    if (!table) {
        alert(window.errorMessages?.noResults || 'No table data available to copy');
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
            alert(window.errorMessages?.copySuccess || 'CSV data copied to clipboard');
        } catch (fallbackErr) {
            alert((window.errorMessages?.copyFailed || 'Failed to copy to clipboard. Please copy manually:') + '\n\n' + csvString);
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
        if (btnDelta === 0.10) {
            btn.classList.add('bg-green-600/40', 'border-2', 'border-green-400', 'hover:bg-green-600/60');
        } else if (btnDelta === 0.20) {
            btn.classList.add('bg-orange-600/40', 'border-2', 'border-orange-400', 'hover:bg-orange-600/60');
        } else if (btnDelta === 0.30) {
            btn.classList.add('bg-red-600/40', 'border-2', 'border-red-400', 'hover:bg-red-600/60');
        }

        // Apply selected styling to the matching button
        if (Math.abs(btnDelta - delta) < 0.001) {
            btn.className = 'risk-btn px-4 py-3 rounded-xl text-sm font-bold transition-all transform duration-300 bg-white text-black border-4 border-white shadow-2xl scale-110 ring-4 ring-white/50';
        }
    });
}

function setDefaultRiskLevel(riskLevel) {
    // Use same delta mapping as backend
    const deltaMap = {
        'LOW': 0.10,
        'MOD': 0.20,
        'HIGH': 0.30
    };
    
    const delta = deltaMap[riskLevel] || 0.10;
    selectedDelta = delta; // Update global variable
    
    // Set visual button state
    const riskButtons = document.querySelectorAll('.risk-btn');
    riskButtons.forEach(btn => {
        const btnDelta = parseFloat(btn.dataset.delta);
        if (btnDelta === delta) {
            btn.classList.add('ring-4', 'ring-white/50');
        } else {
            btn.classList.remove('ring-4', 'ring-white/50');
        }
    });
}

function updateAllModeIndicators(executionMode) {
    // Use execution mode from parameter or backend config (should only be CUDA or CPU, never AUTO)
    const mode = executionMode || (backendConfig?.systemStatus?.computeMode);
    const deviceInfo = backendConfig?.systemStatus?.deviceInfo || '';
    
    // Debug what mode we're getting
    console.log('🔍 Mode received:', mode, 'ExecutionMode param:', executionMode);
    
    // Ensure we never see AUTO mode in web interface
    if (mode === 'AUTO') {
        console.error('❌ AUTO mode should never reach the web interface!');
        return;
    }
    
    // Update footer mode indicator
    const footerModeIndicator = document.getElementById('modeIndicator');
    if (footerModeIndicator) {
        if (mode === 'CUDA') {
            footerModeIndicator.className = 'px-3 py-1 rounded-full font-bold text-black';
            footerModeIndicator.style.backgroundColor = '#FBBF24'; // Force yellow color
            footerModeIndicator.innerHTML = '⚡ ACTIVE: CUDA';
        } else {
            footerModeIndicator.className = 'px-3 py-1 rounded-full font-bold text-black';
            footerModeIndicator.style.backgroundColor = '#4ADE80'; // Force green color  
            footerModeIndicator.innerHTML = '🔧 ACTIVE: CPU';
        }
    }
    
    // Update header mode indicator by ID
    const headerModeIndicator = document.getElementById('headerModeIndicator');
    if (headerModeIndicator) {
        if (mode === 'CUDA') {
            headerModeIndicator.className = 'px-3 py-1 rounded-full font-bold text-black bg-yellow-400';
            headerModeIndicator.innerHTML = '⚡ ACTIVE: CUDA';
        } else {
            headerModeIndicator.className = 'px-3 py-1 rounded-full font-bold text-black bg-green-400';
            headerModeIndicator.innerHTML = '🔧 ACTIVE: CPU';
        }
    }
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
        if (btnDelta === 0.10) {
            btn.className = 'risk-btn px-6 py-4 rounded-lg text-sm font-bold text-white bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200';
        } else if (btnDelta === 0.20) {
            btn.className = 'risk-btn px-6 py-4 rounded-lg text-sm font-bold text-white bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-500 hover:to-orange-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200';
        } else if (btnDelta === 0.30) {
            btn.className = 'risk-btn px-6 py-4 rounded-lg text-sm font-bold text-white bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200';
        }

        // Apply selected styling to the matching button
        if (Math.abs(btnDelta - delta) < 0.001) {
            btn.classList.add('ring-4', 'ring-blue-400', 'ring-opacity-75', 'scale-105');
        }
    });
}

async function handleSubmit(e) {
    console.log('🚀 handleSubmit called - ALLOWS EMPTY SYMBOLS FOR S&P 500!');
    e.preventDefault();

    // Validate selectedDelta before proceeding
    if (isNaN(selectedDelta) || selectedDelta <= 0) {
        selectedDelta = parseFloat(document.getElementById('selected-delta').value) || 0.10;
    }

    // Clear any existing notifications when starting new analysis
    hideNotification();
    
    // Show loading and hide footer during processing
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    const errorEl = document.getElementById('error-message');
    if (errorEl) errorEl.classList.add('hidden');
    
    // Hide footer when job starts
    const statusFooter = document.getElementById('statusFooter');
    if (statusFooter) {
        statusFooter.style.display = 'none';
    }

    // Get form values
    const symbolsValue = document.getElementById('symbols').value;
    const expirationValue = document.getElementById('expiration-date').value;
    const cashValue = document.getElementById('available-cash').value;

    // Allow empty symbols - backend will use S&P 500 when empty
    // No validation needed for symbols

    if (!expirationValue) {
        		showError(window.errorMessages?.noExpiration || 'Please select an expiration date');
        document.getElementById('loading').classList.add('hidden');
        return;
    }

    // Parse symbols into array (allow empty for S&P 500)
    const symbolsArray = symbolsValue.split('\n')
        .map(s => s.trim().toUpperCase())
        .filter(s => s.length > 0);
    
	// Create JSON request data
	const requestData = {
		symbols: symbolsArray,
		expiration_date: expirationValue,
		target_delta: selectedDelta,
		available_cash: parseFloat(cashValue),
		strategy: currentStrategy
	};
	
	// Store for CSV download
	window.lastAnalysisRequest = requestData;
	
	try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error('HTTP error! status: ' + response.status + ', message: ' + errorText);
        }

        // Get the response text first to debug JSON parsing issues
        const responseText = await response.text();
        console.log('📡 Response size:', responseText.length, 'characters');
        
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (jsonError) {
            console.error('❌ JSON Parse Error:', jsonError.message);
            console.error('📄 Response preview (first 500 chars):', responseText.substring(0, 500));
            console.error('📄 Response ending (last 500 chars):', responseText.substring(responseText.length - 500));
            throw new Error('JSON parsing failed: ' + jsonError.message + ' (Response size: ' + responseText.length + ' chars)');
        }
        
        displayResults(data);
        
        // Analysis complete - download button is now available (no notification needed)

    } catch (error) {
        // Clear notification on error
        hideNotification();
        
        showError((window.errorMessages?.analysisError || 'Analysis failed:') + ' ' + error.message);
        
        // Show footer again even on error
        const workloadStatus = document.getElementById('workloadStatus');
        if (workloadStatus) {
            workloadStatus.style.display = 'inline';
            workloadStatus.innerHTML = '🔥 WORKLOAD: ERROR';
        }
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
        selectRiskDeltaQuest(0.10);
    }, 100);
});