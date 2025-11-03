// Modern AXFI Dashboard JavaScript

// Tab switching
function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    // Load watchlists if watchlists tab is shown
    if (tabName === 'watchlists') {
        console.log('Watchlists tab selected, loading watchlists...');
        loadWatchlists();
    }
}

// Theme Toggle
function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    body.setAttribute('data-theme', newTheme);
    
    // Update icon
    const icon = document.getElementById('themeIcon');
    icon.className = newTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    
    // Save preference
    localStorage.setItem('theme', newTheme);
}

// Load saved theme
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.body.setAttribute('data-theme', savedTheme);
    const icon = document.getElementById('themeIcon');
    icon.className = savedTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
});

// Auth Modal
function openAuthModal() {
    document.getElementById('authModal').classList.add('active');
    showLogin();
}

function closeAuthModal() {
    document.getElementById('authModal').classList.remove('active');
}

function showLogin() {
    document.getElementById('loginForm').classList.add('active');
    document.getElementById('registerForm').classList.remove('active');
}

function showRegister() {
    document.getElementById('registerForm').classList.add('active');
    document.getElementById('loginForm').classList.remove('active');
}

function handleLogin() {
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;
    
    // Simple validation (replace with real auth)
    if (username && password) {
        localStorage.setItem('authenticated', 'true');
        localStorage.setItem('username', username);
        updateAuthUI();
        closeAuthModal();
        
        // Show welcome message
        showNotification('Welcome back, ' + username + '!');
    } else {
        showNotification('Please enter username and password', 'error');
    }
}

function handleRegister() {
    const username = document.getElementById('regUsername').value;
    const email = document.getElementById('regEmail').value;
    const password = document.getElementById('regPassword').value;
    
    // Simple validation
    if (username && email && password) {
        localStorage.setItem('authenticated', 'true');
        localStorage.setItem('username', username);
        updateAuthUI();
        closeAuthModal();
        
        showNotification('Account created successfully!', 'success');
    } else {
        showNotification('Please fill all fields', 'error');
    }
}

function handleLogout() {
    localStorage.removeItem('authenticated');
    localStorage.removeItem('username');
    updateAuthUI();
    showNotification('Logged out successfully');
}

function updateAuthUI() {
    const isAuth = localStorage.getItem('authenticated') === 'true';
    const username = localStorage.getItem('username');
    const loginBtn = document.getElementById('loginBtn');
    const userMenu = document.getElementById('userMenu');
    
    if (isAuth) {
        loginBtn.style.display = 'none';
        userMenu.style.display = 'block';
    } else {
        loginBtn.style.display = 'block';
        userMenu.style.display = 'none';
    }
}

// Close modal on outside click
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeAuthModal();
        }
    });
});

// Close modal on X click
document.querySelectorAll('.close-modal').forEach(btn => {
    btn.addEventListener('click', closeAuthModal);
});

// User menu toggle
function toggleUserMenu() {
    const dropdown = document.getElementById('userDropdown');
    dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
}

// AI Chat
function openAIChat() {
    document.getElementById('aiChatSidebar').classList.add('active');
}

function closeAIChat() {
    document.getElementById('aiChatSidebar').classList.remove('active');
}

function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    addChatMessage('user', message);
    input.value = '';
    
    // Simulate AI response
    setTimeout(() => {
        const responses = [
            "Based on current market data, I recommend analyzing AAPL for a potential long position.",
            "The S&P 500 is showing strong momentum. Consider tech sector stocks.",
            "Looking at the data, I'd suggest checking out NVDA - strong technical indicators.",
            "Market volatility is elevated. Consider defensive strategies.",
            "Based on the quantitative analysis, your portfolio is well-diversified."
        ];
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        addChatMessage('bot', randomResponse);
    }, 1000);
}

function addChatMessage(type, message) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    
    if (type === 'bot') {
        messageDiv.innerHTML = `
            <i class="fas fa-robot"></i>
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <i class="fas fa-user"></i>
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Allow Enter key in chat
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }
});

// Notifications
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 2rem;
        background: ${type === 'error' ? '#ef4444' : '#10b981'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        z-index: 4000;
        animation: slideInRight 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Stock Analyzer
async function analyzeSymbol() {
    const symbol = document.getElementById('symbolInput').value.toUpperCase().trim();
    
    if (!symbol) {
        showNotification('Please enter a symbol', 'error');
        return;
    }
    
    const resultDiv = document.getElementById('analyzerResult');
    resultDiv.innerHTML = '<div style="text-align: center; padding: 40px;"><div class="loader"></div><p>Analyzing ' + symbol + '...</p></div>';
    
    try {
        const response = await fetch(`/api/symbol_analysis?symbol=${symbol}`);
        const data = await response.json();
        
        if (data.error) {
            resultDiv.innerHTML = `<div class="error-message"><h2>Error</h2><p>${data.error}</p></div>`;
            return;
        }
        
        const analysis = data.analysis;
        const details = analysis.detailed_info || {};
        
        const formatNumber = (num) => {
            if (!num) return 'N/A';
            if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
            if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
            return num.toFixed(2);
        };
        
        const formatDate = (dateStr) => {
            if (!dateStr) return 'N/A';
            try {
                return new Date(dateStr).toLocaleDateString();
            } catch {
                return dateStr;
            }
        };
        
        const html = `
            <div class="analysis-result">
                <h2><i class="fas fa-chart-line"></i> ${analysis.symbol} - $${analysis.current_price.toFixed(2)}</h2>
                
                <div class="stock-info-card">
                    <h3>Stock Details</h3>
                    <div class="stock-details-grid">
                        <div class="stock-details-col">
                            <div class="detail-row">
                                <span>Previous Close:</span>
                                <strong>${details.previous_close ? details.previous_close.toFixed(2) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Open:</span>
                                <strong>${details.open ? details.open.toFixed(2) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Bid:</span>
                                <strong>N/A</strong>
                            </div>
                            <div class="detail-row">
                                <span>Ask:</span>
                                <strong>N/A</strong>
                            </div>
                            <div class="detail-row">
                                <span>Day's Range:</span>
                                <strong>${details.day_low ? details.day_low.toFixed(2) : 'N/A'} - ${details.day_high ? details.day_high.toFixed(2) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>52 Week Range:</span>
                                <strong>${details['52_week_low'] ? details['52_week_low'].toFixed(2) : 'N/A'} - ${details['52_week_high'] ? details['52_week_high'].toFixed(2) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Volume:</span>
                                <strong>${details.volume ? formatNumber(details.volume) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Avg. Volume:</span>
                                <strong>${details.avg_volume ? formatNumber(details.avg_volume) : 'N/A'}</strong>
                            </div>
                        </div>
                        <div class="stock-details-col">
                            <div class="detail-row">
                                <span>Market Cap (intraday):</span>
                                <strong>${details.market_cap ? formatNumber(details.market_cap) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Beta (5Y Monthly):</span>
                                <strong>${details.beta ? details.beta.toFixed(2) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>PE Ratio (TTM):</span>
                                <strong>${details.pe_ratio ? details.pe_ratio.toFixed(2) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>EPS (TTM):</span>
                                <strong>${details.eps ? details.eps.toFixed(2) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Earnings Date:</span>
                                <strong>${details.earnings_date ? formatDate(details.earnings_date) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Forward Dividend & Yield:</span>
                                <strong>${details.dividend_rate ? details.dividend_rate.toFixed(2) : 'N/A'} ${details.dividend_yield ? '(' + (details.dividend_yield * 100).toFixed(2) + '%)' : ''}</strong>
                            </div>
                            <div class="detail-row">
                                <span>Ex-Dividend Date:</span>
                                <strong>${details.ex_dividend_date ? formatDate(details.ex_dividend_date) : 'N/A'}</strong>
                            </div>
                            <div class="detail-row">
                                <span>1y Target Est:</span>
                                <strong>${details.target_price ? details.target_price.toFixed(2) : 'N/A'}</strong>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h3><i class="fas fa-calendar-day"></i> Short-Term (3 days)</h3>
                        <span class="badge ${analysis.short_term.recommendation}">${analysis.short_term.recommendation}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${analysis.short_term.confidence * 100}%"></div>
                        </div>
                        <p><strong>Confidence:</strong> ${(analysis.short_term.confidence * 100).toFixed(0)}%</p>
                        <p><strong>Price Change:</strong> ${(analysis.short_term.price_change_pct || 0).toFixed(2)}%</p>
                        ${analysis.short_term.rsi ? `<p><strong>RSI:</strong> ${analysis.short_term.rsi.toFixed(2)}</p>` : ''}
                        ${analysis.short_term.macd_signal ? `<p><strong>MACD:</strong> ${analysis.short_term.macd_signal}</p>` : ''}
                        ${analysis.short_term.buy_signals && analysis.short_term.buy_signals.length > 0 ? `<p><strong>Buy Signals:</strong> ${analysis.short_term.buy_signals.join(', ')}</p>` : ''}
                    </div>
                    
                    <div class="analysis-card">
                        <h3><i class="fas fa-calendar-week"></i> Mid-Term (4 weeks)</h3>
                        <span class="badge ${analysis.mid_term.recommendation.replace(' ', '_')}">${analysis.mid_term.recommendation}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${analysis.mid_term.confidence * 100}%"></div>
                        </div>
                        <p><strong>Confidence:</strong> ${(analysis.mid_term.confidence * 100).toFixed(0)}%</p>
                        ${analysis.mid_term.trend ? `<p><strong>Trend:</strong> ${analysis.mid_term.trend}</p>` : ''}
                        ${analysis.mid_term.target_price ? `<p><strong>Target:</strong> $${analysis.mid_term.target_price.toFixed(2)}</p>` : ''}
                    </div>
                    
                    <div class="analysis-card">
                        <h3><i class="fas fa-calendar-month"></i> Long-Term (6 months)</h3>
                        <span class="badge ${analysis.long_term.recommendation}">${analysis.long_term.recommendation}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${analysis.long_term.confidence * 100}%"></div>
                        </div>
                        <p><strong>Confidence:</strong> ${(analysis.long_term.confidence * 100).toFixed(0)}%</p>
                        ${analysis.long_term.total_return_pct ? `<p><strong>Return:</strong> ${analysis.long_term.total_return_pct.toFixed(2)}%</p>` : ''}
                        ${analysis.long_term.long_trend ? `<p><strong>Trend:</strong> ${analysis.long_term.long_trend}</p>` : ''}
                    </div>
                </div>
                
                ${analysis.overall_recommendation ? `
                <div class="overall-recommendation">
                    <h3><i class="fas fa-bullseye"></i> Overall Recommendation</h3>
                    <span class="badge ${analysis.overall_recommendation.recommendation.replace(' ', '_')}">${analysis.overall_recommendation.recommendation}</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${analysis.overall_recommendation.confidence * 100}%"></div>
                    </div>
                    <p><strong>Confidence:</strong> ${(analysis.overall_recommendation.confidence * 100).toFixed(0)}%</p>
                    ${analysis.overall_recommendation.reasoning ? `<p>${analysis.overall_recommendation.reasoning}</p>` : ''}
                </div>
                ` : ''}
                
                ${analysis.exit_strategies && analysis.exit_strategies.length > 0 ? `
                <div class="exit-strategies">
                    <h3><i class="fas fa-sign-out-alt"></i> Exit Strategies</h3>
                    <div class="exit-grid">
                        ${analysis.exit_strategies.map(strategy => `
                            <div class="exit-card ${strategy.type.toLowerCase().replace(' ', '-')}">
                                <strong>${strategy.type}</strong>
                                <div class="exit-price">$${strategy.level.toFixed(2)}</div>
                                <p>${strategy.reason}</p>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        
        resultDiv.innerHTML = html;
        showNotification('Analysis complete!');
    } catch (error) {
        resultDiv.innerHTML = `<div class="error-message"><h2>Error</h2><p>${error.message}</p></div>`;
        showNotification('Analysis failed', 'error');
    }
}

function analyzeSymbolFromTable(symbol) {
    showTab('analyzer');
    document.getElementById('symbolInput').value = symbol;
    analyzeSymbol();
}

// Daily Scan
async function runDailyScan() {
    const button = event.target;
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning...';
    
    try {
        const response = await fetch('/api/daily_scan');
        const data = await response.json();
        
        if (data.recommendations && data.recommendations.length > 0) {
            // Store timestamp in localStorage as backup
            if (data.timestamp) {
                localStorage.setItem('lastScanTimestamp', data.timestamp);
            }
            
            // Update timestamp immediately
            const timestampEl = document.getElementById('lastScanTimestamp');
            if (timestampEl && data.timestamp) {
                timestampEl.textContent = formatTimestamp(data.timestamp);
            }
            
            // Reset button before reload
            button.innerHTML = originalText;
            button.disabled = false;
            
            // Wait a moment to ensure cache is saved, then reload
            showNotification(`Scan complete! Found ${data.total_recommendations} recommendations.`);
            setTimeout(() => {
                location.reload();
            }, 1500); // Increased delay to ensure cache is fully written
        } else {
            // Update timestamp even if no recommendations
            const timestampEl = document.getElementById('lastScanTimestamp');
            if (timestampEl && data.timestamp) {
                timestampEl.textContent = formatTimestamp(data.timestamp);
            }
            button.innerHTML = originalText;
            button.disabled = false;
            showNotification(`Found ${data.total_recommendations} recommendations!`);
        }
    } catch (err) {
        console.error('Scan error:', err);
        button.innerHTML = originalText;
        button.disabled = false;
        showNotification('Scan failed', 'error');
    }
}

// Format timestamp to readable format
function formatTimestamp(timestampStr) {
    if (!timestampStr) return 'Never';
    try {
        const date = new Date(timestampStr);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        return timestampStr;
    }
}

// Watchlists
let currentWatchlistName = null;

async function loadWatchlists() {
    try {
        console.log('Fetching watchlists from /api/watchlists...');
        const response = await fetch('/api/watchlists');
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('HTTP error:', response.status, errorText);
            showNotification(`Failed to load watchlists: HTTP ${response.status} - ${errorText}`, 'error');
            return;
        }
        
        const data = await response.json();
        console.log('Watchlists API response:', data);
        
        if (data.status === 'success') {
            const defaultLists = data.default_lists || {};
            const customWatchlists = data.watchlists || [];
            console.log('Default lists:', Object.keys(defaultLists), defaultLists);
            console.log('Custom watchlists:', customWatchlists);
            
            if (Object.keys(defaultLists).length === 0 && customWatchlists.length === 0) {
                console.warn('No watchlists found in response');
                showNotification('No watchlists found. Default lists will be initialized.', 'error');
            }
            
            renderWatchlists(defaultLists, customWatchlists);
        } else {
            console.error('API returned error status:', data);
            const errorMsg = data.message || 'Unknown error';
            showNotification('Failed to load watchlists: ' + errorMsg, 'error');
            // Still try to render what we got
            renderWatchlists(data.default_lists || {}, data.watchlists || []);
        }
    } catch (error) {
        console.error('Error loading watchlists:', error);
        console.error('Error stack:', error.stack);
        showNotification('Failed to load watchlists: ' + (error.message || 'Network error'), 'error');
    }
}

function renderWatchlists(defaultLists, customWatchlists) {
    const container = document.getElementById('watchlistsList');
    if (!container) return;
    
    let html = '<div class="watchlists-grid">';
    
    // Render default lists
    if (defaultLists && typeof defaultLists === 'object') {
        const defaultEntries = Object.entries(defaultLists);
        console.log(`Rendering ${defaultEntries.length} default watchlists`);
        for (const [key, watchlist] of defaultEntries) {
            if (watchlist && watchlist.name) {
                html += renderWatchlistCard(watchlist, true);
            }
        }
    }
    
    // Render custom watchlists
    if (Array.isArray(customWatchlists)) {
        console.log(`Rendering ${customWatchlists.length} custom watchlists`);
        for (const watchlist of customWatchlists) {
            if (watchlist && watchlist.name) {
                html += renderWatchlistCard(watchlist, false);
            }
        }
    }
    
    if (html === '<div class="watchlists-grid">') {
        html += '<div class="empty-state"><i class="fas fa-list"></i><p>No watchlists found</p></div>';
    }
    
    html += '</div>';
    container.innerHTML = html;
}

function renderWatchlistCard(watchlist, isDefault) {
    const symbols = watchlist.symbols || [];
    const readonly = watchlist.readonly || isDefault;
    
    let html = `
        <div class="watchlist-card">
            <div class="watchlist-header">
                <h3>
                    <i class="fas fa-list"></i> ${watchlist.name}
                    ${isDefault ? '<span class="badge badge-default">Default</span>' : ''}
                </h3>
                <div class="watchlist-actions">
                    ${!readonly ? `<button class="icon-btn-small" onclick="deleteWatchlist('${watchlist.name}')" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>` : ''}
                </div>
            </div>
            <div class="watchlist-info">
                <span class="watchlist-count">${symbols.length} symbols</span>
                <button class="btn-small" onclick="showAddSymbolModal('${watchlist.name}')">
                    <i class="fas fa-plus"></i> Add Symbol
                </button>
            </div>
            <div class="watchlist-symbols">
    `;
    
    if (symbols.length > 0) {
        symbols.forEach(symbol => {
            html += `
                <div class="symbol-tag">
                    <span onclick="analyzeSymbolFromTable('${symbol}')">${symbol}</span>
                    ${!readonly ? `<button class="icon-btn-tiny" onclick="removeSymbolFromWatchlist('${watchlist.name}', '${symbol}')" title="Remove">
                        <i class="fas fa-times"></i>
                    </button>` : ''}
                </div>
            `;
        });
    } else {
        html += '<p class="empty-watchlist">No symbols in this watchlist</p>';
    }
    
    html += `
            </div>
        </div>
    `;
    
    return html;
}

function showCreateWatchlistModal() {
    document.getElementById('createWatchlistModal').classList.add('active');
}

function closeCreateWatchlistModal() {
    document.getElementById('createWatchlistModal').classList.remove('active');
    document.getElementById('newWatchlistName').value = '';
}

async function createWatchlist() {
    const name = document.getElementById('newWatchlistName').value.trim();
    
    if (!name) {
        showNotification('Please enter a watchlist name', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/watchlists/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: name, symbols: [] })
        });
        
        const data = await response.json();
        console.log('Create watchlist response:', data);
        
        if (data.status === 'success') {
            showNotification(data.message || 'Watchlist created successfully', 'success');
            closeCreateWatchlistModal();
            loadWatchlists();
        } else {
            showNotification(data.message || 'Failed to create watchlist', 'error');
        }
    } catch (error) {
        console.error('Error creating watchlist:', error);
        showNotification('Failed to create watchlist: ' + error.message, 'error');
    }
}

async function deleteWatchlist(name) {
    if (!confirm(`Are you sure you want to delete the watchlist "${name}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/watchlists/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification(data.message, 'success');
            loadWatchlists();
        } else {
            showNotification(data.message, 'error');
        }
    } catch (error) {
        console.error('Error deleting watchlist:', error);
        showNotification('Failed to delete watchlist', 'error');
    }
}

function showAddSymbolModal(watchlistName) {
    currentWatchlistName = watchlistName;
    document.getElementById('addSymbolWatchlistName').textContent = `Adding to: ${watchlistName}`;
    document.getElementById('addSymbolModal').classList.add('active');
}

function closeAddSymbolModal() {
    document.getElementById('addSymbolModal').classList.remove('active');
    document.getElementById('addSymbolInput').value = '';
    currentWatchlistName = null;
}

async function addSymbolToWatchlist() {
    const symbol = document.getElementById('addSymbolInput').value.trim().toUpperCase();
    
    if (!symbol) {
        showNotification('Please enter a symbol', 'error');
        return;
    }
    
    if (!currentWatchlistName) {
        showNotification('No watchlist selected', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/watchlists/${encodeURIComponent(currentWatchlistName)}/add`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol: symbol })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification(data.message, 'success');
            closeAddSymbolModal();
            loadWatchlists();
        } else {
            showNotification(data.message, 'error');
        }
    } catch (error) {
        console.error('Error adding symbol:', error);
        showNotification('Failed to add symbol', 'error');
    }
}

async function removeSymbolFromWatchlist(watchlistName, symbol) {
    try {
        const response = await fetch(`/api/watchlists/${encodeURIComponent(watchlistName)}/remove/${symbol}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification(data.message, 'success');
            loadWatchlists();
        } else {
            showNotification(data.message, 'error');
        }
    } catch (error) {
        console.error('Error removing symbol:', error);
        showNotification('Failed to remove symbol', 'error');
    }
}

// Close modals on outside click
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            if (modal.id === 'createWatchlistModal') {
                closeCreateWatchlistModal();
            } else if (modal.id === 'addSymbolModal') {
                closeAddSymbolModal();
            }
        }
    });
});

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    updateAuthUI();
    
    // Format timestamp if present (always format, even if "Never")
    const timestampEl = document.getElementById('lastScanTimestamp');
    if (timestampEl) {
        let timestampText = timestampEl.textContent.trim();
        
        // Check localStorage for backup timestamp if template shows "Never"
        if ((timestampText === 'Never' || !timestampText) && localStorage.getItem('lastScanTimestamp')) {
            timestampText = localStorage.getItem('lastScanTimestamp');
            timestampEl.textContent = formatTimestamp(timestampText);
        } else if (timestampText && timestampText !== 'Never' && timestampText !== '') {
            // Only format if it looks like a timestamp (contains 'T' or is ISO format)
            if (timestampText.includes('T') || timestampText.match(/\d{4}-\d{2}-\d{2}/)) {
                timestampEl.textContent = formatTimestamp(timestampText);
                // Also save to localStorage
                localStorage.setItem('lastScanTimestamp', timestampText);
            }
        }
    }
    
    // Allow Enter key in watchlist modals
    const newWatchlistInput = document.getElementById('newWatchlistName');
    if (newWatchlistInput) {
        newWatchlistInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                createWatchlist();
            }
        });
    }
    
    const addSymbolInput = document.getElementById('addSymbolInput');
    if (addSymbolInput) {
        addSymbolInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                addSymbolToWatchlist();
            }
        });
    }
    
    // Load watchlists when watchlists tab is shown
    const watchlistsTab = document.getElementById('watchlists');
    if (watchlistsTab) {
        // Also check if tab is already active on load
        if (watchlistsTab.classList.contains('active')) {
            console.log('Watchlists tab is active on page load, loading watchlists...');
            loadWatchlists();
        }
        
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const target = mutation.target;
                    if (target.classList.contains('active')) {
                        console.log('Watchlists tab became active, loading watchlists...');
                        loadWatchlists();
                    }
                }
            });
        });
        observer.observe(watchlistsTab, { attributes: true });
    } else {
        console.error('Watchlists tab element not found!');
    }
});

