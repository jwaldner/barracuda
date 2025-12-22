/*
 * CRITICAL RULES - NEVER VIOLATE:
 * 1. NO hardcoded strings, arrays, or config values in this file
 * 2. ALL data comes from window.templateFunction() from backend
 * 3. Config changes in YAML should never require JS changes
 * 4. Use backend template functions for ALL dynamic content
 */

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
            
            // Make template functions available globally
            window.tableFieldKeys = function() {
                return backendConfig.tableFieldKeys || [];
            };
            window.tableHeaders = function() {
                return backendConfig.tableHeaders || [];
            };
            window.generateCSVFilename = function(targetDelta, expirationDate, strategy, symbolCount) {
                // Generate filename using config pattern - if available from backend
                const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false }).replace(/:/g, '-');
                return `${timestamp}_${expirationDate}_delta${targetDelta.toFixed(2)}_${strategy}_${symbolCount}.csv`;
            };
            
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
    
	// Standard config utilities are provided by template - no setup needed    // Setup risk selector event listeners
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
            if (!window.lastResults || !window.lastResults.length) {
                showNotification('❌ No analysis data available. Run analysis first.', 'error');
                return;
            }
            
            const headers = window.csvHeaders || [];
            if (!headers.length) {
                showNotification('❌ CSV headers not available. Please refresh the page.', 'error');
                return;
            }
            
            let csvContent = headers.join(',') + '\n';
            
            // Helper function to get raw value for CSV
            const getRawValue = (field) => {
                if (!field) return '';
                if (typeof field === 'object' && field.raw !== undefined) {
                    return field.raw;
                }
                return field;
            };
            
            // Get field mapping from backend - NO hardcoded fields
            const fieldKeys = window.tableFieldKeys ? window.tableFieldKeys() : [];
            if (!fieldKeys.length) {
                showNotification('❌ Field mapping not available. Please refresh the page.', 'error');
                return;
            }
            
            window.lastResults.forEach((option, index) => {
                const row = [];
                
                // Build row dynamically using backend field mapping
                fieldKeys.forEach(fieldKey => {
                    let value = getRawValue(option[fieldKey]);
                    
                    // Handle special cases
                    if (fieldKey === 'rank' && !value) {
                        value = index + 1;
                    }
                    if (fieldKey === 'company' && !value) {
                        value = getRawValue(option.ticker) || '';
                    }
                    if (fieldKey === 'sector' && !value) {
                        value = 'Unknown';
                    }
                    
                    // Quote values that contain commas
                    if (typeof value === 'string' && value.includes(',')) {
                        value = `"${value}"`;
                    }
                    
                    row.push(value || '');
                });
                
                csvContent += row.join(',') + '\n';
            });
            
            // Copy to clipboard with error handling
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(csvContent).then(function() {
                    showNotification('📋 CSV copied to clipboard!', 'success');
                }).catch(function(err) {
                    console.error('Clipboard copy failed:', err);
                    showNotification('❌ Failed to copy to clipboard. Browser may not support this feature.', 'error');
                });
            } else {
                showNotification('❌ Clipboard not supported in this browser.', 'error');
            }
        });
    }
    
    // CSV Download functionality (Download file) - LIGHTNING FAST using cached data
    const downloadCSVBtn = document.getElementById('downloadCSVBtn');
    if (downloadCSVBtn) {
        downloadCSVBtn.addEventListener('click', function() {
            if (!window.lastResults || !window.lastAnalysisRequest) {
                showNotification('❌ No analysis data available for download. Run analysis first.', 'error');
                return;
            }
            
            // ⚡ Generate CSV from cached data instantly - NO API CALLS!
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
            
            // Get field mapping from backend - NO hardcoded fields
            const fieldKeys = window.tableFieldKeys() || [];
            
            window.lastResults.forEach((option, index) => {
                const row = [];
                
                // Build row dynamically using backend field mapping
                fieldKeys.forEach(fieldKey => {
                    let value = getRawValue(option[fieldKey]);
                    
                    // Handle special cases
                    if (fieldKey === 'rank' && !value) {
                        value = index + 1;
                    }
                    if (fieldKey === 'company' && !value) {
                        value = getRawValue(option.ticker) || '';
                    }
                    if (fieldKey === 'sector' && !value) {
                        value = 'Unknown';
                    }
                    
                    // Quote values that contain commas
                    if (typeof value === 'string' && value.includes(',')) {
                        value = `"${value}"`;
                    }
                    
                    row.push(value || '');
                });
                
                csvContent += row.join(',') + '\n';
            });
            
            // Use template function directly - standard pattern
            const filename = window.generateCSVFilename(
                window.lastAnalysisRequest.target_delta || 0.30,
                window.lastAnalysisRequest.expiration_date || 'unknown',
                window.lastAnalysisRequest.strategy || 'options',
                window.lastResults.length
            );
            
            // ⚡ Instant download - no network request!
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Store for potential reuse
            window.lastCSVBlob = blob;
            
            // Clean up
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 1000);
            
            // Show sticky download notification with generic message
            showStickyDownloadNotification();
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
        
        // Build row dynamically using backend field mapping to match headers
        const fieldKeys = window.tableFieldKeys() || [];
        let rowHTML = '';
        
        fieldKeys.forEach(fieldKey => {
            let value = '';
            let cssClass = 'px-6 py-4 whitespace-nowrap text-sm';
            
            switch(fieldKey) {
                case 'rank':
                    value = getValue(option.rank) || (index + 1);
                    cssClass += ' font-bold text-gray-900';
                    break;
                case 'ticker':
                    value = getValue(option.ticker);
                    cssClass += ' font-bold text-gray-900';
                    break;
                case 'company':
                    value = companyName;
                    cssClass = 'company-tooltip px-6 py-4 whitespace-nowrap text-sm text-gray-700';
                    rowHTML += `<td class="${cssClass}" data-sector="${sectorName}">${value}</td>`;
                    return;
                case 'sector':
                    value = sectorName;
                    cssClass = 'sector-column px-6 py-4 whitespace-nowrap text-sm text-gray-600';
                    break;
                case 'strike':
                    value = getValue(option.strike);
                    cssClass = getCSSClass(option.strike, cssClass);
                    break;
                case 'stock_price':
                    value = getValue(option.stock_price);
                    cssClass = getCSSClass(option.stock_price, cssClass);
                    break;
                case 'premium':
                    value = getValue(option.premium);
                    cssClass = getCSSClass(option.premium, cssClass);
                    break;
                case 'max_contracts':
                    value = getValue(option.max_contracts);
                    cssClass = getCSSClass(option.max_contracts, cssClass);
                    break;
                case 'total_premium':
                    value = getValue(option.total_premium);
                    cssClass = getCSSClass(option.total_premium, cssClass + ' font-bold');
                    break;
                case 'profit_percentage':
                    value = getValue(option.profit_percentage);
                    cssClass = getCSSClass(option.profit_percentage, cssClass + ' font-bold');
                    break;
                case 'delta':
                    value = getValue(option.delta);
                    cssClass = getCSSClass(option.delta, cssClass);
                    break;
                case 'expiration':
                    value = getValue(option.expiration);
                    cssClass = getCSSClass(option.expiration, cssClass + ' text-gray-500');
                    break;
                case 'days_to_exp':
                    value = getValue(option.days_to_exp);
                    cssClass = getCSSClass(option.days_to_exp, cssClass);
                    break;
                default:
                    value = getValue(option[fieldKey]) || '';
            }
            
            rowHTML += `<td class="${cssClass}">${value}</td>`;
        });
        
        row.innerHTML = rowHTML;
        
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
        const computeTime = data.meta.compute_duration ? 
            (data.meta.compute_duration < 0.001 ? 
                (data.meta.compute_duration * 1000).toFixed(2) + 'ms' : 
                data.meta.compute_duration.toFixed(3) + 's') : '0.000s';
        let footerContent = `🔥 ${executionMode} | ⏰ ${processingTime}s TOTAL | 🔋 ${computeTime} ${executionMode} COMPUTE 🔋 | 📊 ${totalRecords.toLocaleString()} RECORDS`;
        
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
        console.error('No table found with selector "#results table"');
        alert(window.errorMessages?.noResults || 'No table data available to copy');
        return;
    }

    console.log('Found table:', table);
    const rows = table.querySelectorAll('tr');
    console.log('Found rows:', rows.length);
    
    if (rows.length === 0) {
        alert('No data rows found in table');
        return;
    }

    const csvContent = [];

    // Process each row
    rows.forEach((row, rowIndex) => {
        console.log(`Processing row ${rowIndex}:`, row);
        const cells = row.querySelectorAll('th, td');
        console.log(`Row ${rowIndex} has ${cells.length} cells`);
        const rowData = [];
        
        cells.forEach((cell, cellIndex) => {
            // Get text content and clean it up
            let text = cell.textContent ? cell.textContent.trim() : '';
            console.log(`Cell ${cellIndex} raw content: "${cell.textContent}"`);
            console.log(`Cell ${cellIndex} cleaned content: "${text}"`);
            
            // Clean the text but keep it simple for Google Sheets
            text = text.replace(/\s+/g, ' ').replace(/\t/g, ' ').replace(/\n/g, ' ').replace(/\r/g, ' ');
            
            // Always add the cell content to maintain column structure
            rowData.push(text);
        });
        
        console.log(`Row ${rowIndex} final data array:`, rowData);
        console.log(`Row ${rowIndex} joined with tabs: "${rowData.join('\t')}"`);
        
        // Always add the row to maintain consistent structure
        if (rowData.length > 0) {
            csvContent.push(rowData.join('\t'));
        }
    });
    
    console.log('Final csvContent array:', csvContent);

    console.log('Final csvContent array:', csvContent);

    const csvString = csvContent.join('\n');
    
    console.log('Raw TSV string:', csvString);
    
    // For Google Sheets: Convert to proper CSV format as per RFC 4180
    // Google Sheets expects commas, proper quoting, and escaped quotes
    const properCSV = csvContent.map((row, rowIndex) => {
        console.log(`Converting row ${rowIndex}: "${row}"`);
        const fields = row.split('\t');
        console.log(`Row ${rowIndex} split into fields:`, fields);
        
        // Remove empty trailing fields
        while (fields.length > 0 && fields[fields.length - 1].trim() === '') {
            fields.pop();
        }
        console.log(`Row ${rowIndex} after removing trailing empties:`, fields);
        
        // Convert each field to proper CSV format
        const csvFields = fields.map((field, fieldIndex) => {
            // Always trim the field
            field = field.trim();
            console.log(`Field ${fieldIndex}: "${field}"`);
            
            // If field contains comma, quote, or newline - wrap in quotes and escape quotes
            if (field.includes(',') || field.includes('"') || field.includes('\n') || field.includes('\r')) {
                // Escape existing quotes by doubling them
                field = field.replace(/"/g, '""');
                // Wrap in quotes
                const quotedField = `"${field}"`;
                console.log(`Field ${fieldIndex} needs quoting: "${quotedField}"`);
                return quotedField;
            }
            
            return field;
        });
        
        const csvRow = csvFields.join(',');
        console.log(`Row ${rowIndex} final CSV: "${csvRow}"`);
        return csvRow;
    }).join('\n');
    
    console.log('=== FINAL CSV OUTPUT ===');
    console.log(properCSV);
    console.log('=== END CSV OUTPUT ===');

    try {
        // For Google Sheets paste: Use simple format without quotes unless absolutely necessary
        // Google Sheets paste is more forgiving with simple comma separation
        const googleSheetsCSV = csvContent.map(row => {
            const fields = row.split('\t');
            // Remove empty trailing fields
            while (fields.length > 0 && fields[fields.length - 1].trim() === '') {
                fields.pop();
            }
            
            // For Google Sheets paste: minimal quoting, just commas
            return fields.map(field => {
                field = field.trim();
                // Only quote if field actually contains a comma
                if (field.includes(',')) {
                    return `"${field.replace(/"/g, '""')}"`;
                }
                return field;
            }).join(',');
        }).join('\n');
        
        console.log('Google Sheets paste format:', googleSheetsCSV);
        
        // Copy the Google Sheets optimized format
        await navigator.clipboard.writeText(googleSheetsCSV);
        
        // Use the same format for file download
        // Add UTF-8 BOM for better Google Sheets compatibility
        const utf8BOM = '\uFEFF';
        const csvWithBOM = utf8BOM + properCSV;
        
        // Create downloadable link that opens instead of saves
        const csvBlob = new Blob([csvWithBOM], { type: 'text/csv;charset=utf-8' });
        const csvUrl = URL.createObjectURL(csvBlob);
        
        // Create temporary download link
        const downloadLink = document.createElement('a');
        downloadLink.href = csvUrl;
        downloadLink.download = `barracuda_analysis_${new Date().toISOString().slice(0,19).replace(/[:.]/g, '-')}.csv`;
        downloadLink.target = '_blank';
        
        // Trigger download which will open in browser
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        
        // Clean up the URL after a delay
        setTimeout(() => URL.revokeObjectURL(csvUrl), 1000);
        
        // Visual feedback
        const btn = document.getElementById('copy-csv-btn');
        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '✅ Downloaded!';
            btn.classList.add('bg-green-500/40');
            
            setTimeout(() => {
                const btnTimeout = document.getElementById('copy-csv-btn');
                if (btnTimeout) {
                    btnTimeout.innerHTML = originalText;
                    btnTimeout.classList.remove('bg-green-500/40');
                }
            }, 2000);
        }
        
        showStickyDownloadNotification();
        
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

// Audit system - ticker input sets audit target, Grok button sends to AI
document.addEventListener('DOMContentLoaded', function() {
    const grokBtn = document.getElementById('grok-btn');
    const auditInput = document.getElementById('audit-ticker-input');
    
    // Check audit file exists and update Grok button state
    window.updateGrokButtonState = async function() {
        if (!grokBtn) return;
        
        try {
            const response = await fetch('/api/audit-exists');
            const result = await response.json();
            console.log('🔍 Audit file exists:', result.exists, '| Button will be:', result.exists ? 'ENABLED' : 'DISABLED');
            grokBtn.disabled = !result.exists;
            
            if (result.exists) {
                grokBtn.title = "Send audit data to Grok AI - Powered by xAI";
            } else {
                grokBtn.title = "No audit data available - run analysis with ticker first";
            }
        } catch (error) {
            console.log('Failed to check audit file existence:', error);
            grokBtn.disabled = true;
        }
    }
    
    // Initial check
    window.updateGrokButtonState();
    
    // Check periodically for file existence
    setInterval(window.updateGrokButtonState, 2000);
    
    // Check if ticker is already set for audit on startup
    if (auditInput && auditInput.value.trim()) {
        const ticker = auditInput.value.trim().toUpperCase();
        console.warn(`⚠️ STARTUP: Audit ticker "${ticker}" is set for detailed logging`);
        // Send startup notification to server log
        fetch('/api/audit-startup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        }).catch(err => console.log('Startup audit notification failed:', err));
        // Initialize audit data for this ticker (overwrites any previous)
        window.auditData = {};
    }
    
    if (grokBtn) {
        grokBtn.addEventListener('click', handleGrokAnalysis);
    }
    
    // Clear audit data when ticker changes (COST CONTROL)
    if (auditInput) {
        let lastTicker = auditInput.value.trim().toUpperCase();
        
        auditInput.addEventListener('input', function() {
            const currentTicker = auditInput.value.trim().toUpperCase();
            
            // If ticker changed, clear ALL audit data (overwrite)
            if (currentTicker !== lastTicker) {
                if (lastTicker) {
                    console.log(`🧹 Clearing all audit data (switched from ${lastTicker} to ${currentTicker})`);
                }
                // Overwrite - clear ALL previous audit data
                window.auditData = {};
                lastTicker = currentTicker;
            }
        });
    }
});

function handleGrokAnalysis() {
    const auditInput = document.getElementById('audit-ticker-input');
    const grokBtn = document.getElementById('grok-btn');
    
    if (!auditInput || !grokBtn) return;
    
    const ticker = auditInput.value.trim().toUpperCase();
    if (!ticker) {
        showNotification('❌ Please enter a ticker to audit first', 'error');
        return;
    }
    
    // Disable button during processing
    grokBtn.disabled = true;
    
    // Read audit data from JSON file via server
    sendToGrok(ticker, null, grokBtn);
}

async function sendToGrok(ticker, auditData, grokBtn) {
    let progressModal = null;
    let startTime = Date.now();
    
    try {
        // Show progress modal with animated spinner
        progressModal = showGrokProgressModal(ticker);
        
        // Enhanced timeout handling with progress updates
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
        
        // Progress tracking interval
        let elapsed = 0;
        const progressInterval = setInterval(() => {
            elapsed = Math.floor((Date.now() - startTime) / 1000);
            updateGrokProgress(progressModal, elapsed, ticker);
        }, 1000);
        
        const response = await fetch('/api/ai-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        clearInterval(progressInterval);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Grok analysis failed (${response.status}): ${errorText}`);
        }
        
        const result = await response.json();
        
        // Close progress modal and show results
        closeGrokProgressModal(progressModal);
        
        // Show analysis in modal instead of just notification
        showGrokAnalysisModal(ticker, result.analysis, elapsed);
        
        // Append to audit log
        await appendAIAnalysisToLog(ticker, result.analysis);
        
        showNotification(`✅ Grok analysis complete for ${ticker}! (${elapsed}s)`, 'success');
        
    } catch (error) {
        if (progressModal) {
            closeGrokProgressModal(progressModal);
        }
        
        console.error('Grok analysis failed:', error);
        
        let errorMsg = error.message;
        if (error.name === 'AbortError') {
            errorMsg = 'Request timed out after 5 minutes. Grok may be experiencing high load.';
        } else if (error.message.includes('fetch')) {
            errorMsg = 'Network error. Check internet connection and try again.';
        }
        
        showNotification(`❌ Grok analysis failed: ${errorMsg}`, 'error');
        
        // Show detailed error modal
        showGrokErrorModal(ticker, errorMsg, Math.floor((Date.now() - startTime) / 1000));
        
    } finally {
        grokBtn.disabled = false;
    }
}

// Grok Progress Modal Functions
function showGrokProgressModal(ticker) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg p-6 max-w-md mx-4 shadow-2xl">
            <div class="text-center">
                <div class="mb-4">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                </div>
                <h3 class="text-lg font-bold text-gray-800 mb-2">🤖 Grok AI Analysis</h3>
                <p class="text-gray-600 mb-4">Analyzing <strong>${ticker}</strong> options data...</p>
                <div class="bg-gray-100 rounded p-3 text-sm">
                    <div id="grok-status">📡 Connecting to xAI servers...</div>
                    <div id="grok-timer" class="text-xs text-gray-500 mt-2">Elapsed: 0s</div>
                </div>
                <div class="mt-4 text-xs text-gray-400">
                    ⏱️ This may take 30-90 seconds for complex analysis<br>
                    💡 Grok is processing Black-Scholes calculations
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    return modal;
}

function updateGrokProgress(modal, elapsed, ticker) {
    const statusEl = modal.querySelector('#grok-status');
    const timerEl = modal.querySelector('#grok-timer');
    
    let status = '📡 Connecting to xAI servers...';
    if (elapsed > 3) status = '🧮 Processing Black-Scholes calculations...';
    if (elapsed > 10) status = '🔍 Analyzing option Greeks and pricing...';
    if (elapsed > 20) status = '📊 Validating mathematical accuracy...';
    if (elapsed > 40) status = '📝 Generating detailed analysis report...';
    if (elapsed > 60) status = '⏳ Finalizing comprehensive review...';
    
    if (statusEl) statusEl.textContent = status;
    if (timerEl) timerEl.textContent = `Elapsed: ${elapsed}s`;
}

function closeGrokProgressModal(modal) {
    if (modal && modal.parentNode) {
        modal.parentNode.removeChild(modal);
    }
}

function showGrokAnalysisModal(ticker, analysis, elapsed) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg max-w-4xl max-h-[90vh] mx-4 shadow-2xl flex flex-col">
            <div class="p-6 border-b flex justify-between items-center">
                <h3 class="text-xl font-bold text-gray-800">🤖 Grok AI Analysis - ${ticker}</h3>
                <div class="flex items-center gap-4">
                    <span class="text-sm text-gray-500">⏱️ ${elapsed}s</span>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 text-2xl">&times;</button>
                </div>
            </div>
            <div class="flex-1 overflow-y-auto p-6">
                <div class="prose max-w-none" id="grok-analysis-content">
                    ${formatAnalysisAsHTML(analysis)}
                </div>
            </div>
            <div class="p-4 border-t bg-gray-50 flex justify-between items-center">
                <div class="text-sm text-gray-600">
                    💾 Analysis saved to audit logs
                </div>
                <div class="flex gap-2">
                    <button onclick="copyGrokAnalysis('${ticker}')" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">📋 Copy</button>
                    <button onclick="this.closest('.fixed').remove()" class="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600">Close</button>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function showGrokErrorModal(ticker, errorMsg, elapsed) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg max-w-md mx-4 shadow-2xl">
            <div class="p-6">
                <div class="text-center">
                    <div class="text-red-500 text-4xl mb-4">❌</div>
                    <h3 class="text-lg font-bold text-gray-800 mb-2">Grok Analysis Failed</h3>
                    <p class="text-gray-600 mb-4">Ticker: <strong>${ticker}</strong></p>
                    <div class="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700 mb-4">
                        ${errorMsg}
                    </div>
                    <div class="text-xs text-gray-500 mb-4">Duration: ${elapsed}s</div>
                    <div class="text-xs text-gray-400 mb-4">
                        💡 <strong>Troubleshooting:</strong><br>
                        • Check if ticker has recent audit data<br>
                        • Verify API key configuration<br>
                        • Try again in a few minutes<br>
                        • Check network connectivity
                    </div>
                </div>
                <button onclick="this.closest('.fixed').remove()" class="w-full bg-gray-500 text-white py-2 rounded hover:bg-gray-600">Close</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function formatAnalysisAsHTML(markdownText) {
    // Convert markdown to HTML (basic conversion)
    return markdownText
        .replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold mb-4">$1</h1>')
        .replace(/^## (.*$)/gm, '<h2 class="text-xl font-semibold mb-3 mt-6">$1</h2>')
        .replace(/^### (.*$)/gm, '<h3 class="text-lg font-medium mb-2 mt-4">$1</h3>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n\n/g, '</p><p class="mb-4">')
        .replace(/^(?!<[h|p])(.+)$/gm, '<p class="mb-4">$1</p>')
        .replace(/^- (.+)$/gm, '<li class="ml-4">• $1</li>');
}

function copyGrokAnalysis(ticker) {
    const content = document.getElementById('grok-analysis-content');
    if (content) {
        const text = content.innerText || content.textContent;
        navigator.clipboard.writeText(text).then(() => {
            showNotification('📋 Analysis copied to clipboard!', 'success');
        }).catch(() => {
            showNotification('❌ Failed to copy to clipboard', 'error');
        });
    }
}

async function appendAIAnalysisToLog(ticker, analysis) {
    const logEntry = {
        timestamp: new Date().toISOString(),
        ticker: ticker,
        ai_analysis: analysis,
        type: 'grok_analysis'
    };
    
    const response = await fetch('/api/audit-log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logEntry)
    });
    
    if (!response.ok) {
        throw new Error(`Failed to create audit file: ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log(`📄 Audit file created: ${result.filename}`);
    
    return result;
}

async function startAuditProcess(ticker, rank, controls) {
    try {
        showNotification(`🔍 Starting audit for ${ticker} (Rank #${rank})...`, 'info');
        
        // Step 1: Gather data for this ticker
        showNotification(`📊 Gathering data for ${ticker}...`, 'info');
        const auditData = await gatherTickerData(ticker, rank);
        
        // Step 2: Send to AI for analysis
        showNotification(`🤖 AI analyzing ${ticker}...`, 'info');
        const aiAnalysis = await getAIAnalysis(auditData);
        
        // Step 3: Append to audit log
        showNotification(`💾 Saving audit log for ${ticker}...`, 'info');
        await appendToAuditLog(auditData, aiAnalysis);
        
        // Step 4: Clear input and re-enable (COST CONTROL)
        if (controls.input) {
            controls.input.value = '';
            controls.input.disabled = false;
            controls.input.focus();
        }
        showNotification(`✅ Audit complete for ${ticker}! Input cleared.`, 'success');
        
    } catch (error) {
        console.error('Audit process failed:', error);
        if (controls.input) {
            controls.input.disabled = false;
        }
        showNotification(`❌ Audit failed for ${ticker}: ${error.message}`, 'error');
    }
}

async function gatherTickerData(ticker, rank) {
    // Check if we have analysis data to audit
    if (!window.lastAnalysisRequest) {
        throw new Error('No analysis data available. Please run an analysis first.');
    }
    
    // Gather all data for this ticker
    const requestData = Object.assign({}, window.lastAnalysisRequest);
    requestData.audit_ticker = ticker;
    requestData.target_delta = selectedDelta;
    
    const response = await fetch('/api/audit-ticker', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
        throw new Error(`Failed to gather data: ${response.statusText}`);
    }
    
    return await response.json();
}

async function getAIAnalysis(auditData) {
    const response = await fetch('/api/ai-analysis', {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(auditData)
    });
    
    if (!response.ok) {
        throw new Error(`AI analysis failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result.analysis;
}

async function appendToAuditLog(auditData, aiAnalysis) {
    const logEntry = {
        timestamp: new Date().toISOString(),
        ticker: auditData.ticker,
        rank: auditData.rank,
        audit_data: auditData,
        ai_analysis: aiAnalysis
    };
    
    const response = await fetch('/api/audit-log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logEntry)
    });
    
    if (!response.ok) {
        throw new Error(`Failed to save audit log: ${response.statusText}`);
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
	
	// Add audit ticker if set
	const auditInput = document.getElementById('audit-ticker-input');
	if (auditInput && auditInput.value.trim()) {
		requestData.audit_ticker = auditInput.value.trim().toUpperCase();
	}
	
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
        
        // Update Grok button state after analysis (audit file may have been created)
        if (window.updateGrokButtonState) {
            window.updateGrokButtonState();
        }
        
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