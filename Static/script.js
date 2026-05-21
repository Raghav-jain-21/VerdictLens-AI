const tabs = ['alimony', 'divorce', 'custody', 'maintenance', 'other', 'dashboard', 'news'];
let isLoggedIn = false;
const STORAGE_KEY = 'verdict_lens_db'; // Key for browser storage

// --- APP LOGIC ---
window.onload = function() {
    switchTab('news');
    updateAuthUI();
};

// --- AUTHENTICATION ---
function openLoginModal() { document.getElementById('login-modal').classList.remove('hidden'); toggleAuthMode('login'); }
function closeLoginModal() { document.getElementById('login-modal').classList.add('hidden'); }

function toggleAuthMode(mode) {
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    const title = document.getElementById('auth-title');
    
    if(mode === 'signup') {
        loginForm.classList.add('hidden');
        signupForm.classList.remove('hidden');
        title.innerText = "New Registration";
    } else {
        signupForm.classList.add('hidden');
        loginForm.classList.remove('hidden');
        title.innerText = "Attorney Login";
    }
}

function handleLogin(e) { e.preventDefault(); isLoggedIn = true; closeLoginModal(); updateAuthUI(); alert("Logged in successfully."); }
function handleSignup(e) { e.preventDefault(); isLoggedIn = true; closeLoginModal(); updateAuthUI(); alert("Account created."); }
function handleLogout() { isLoggedIn = false; updateAuthUI(); }

function updateAuthUI() {
    const loginBtn = document.getElementById('btn-login');
    const userProfile = document.getElementById('user-profile');
    if (isLoggedIn) { loginBtn.classList.add('hidden'); userProfile.classList.remove('hidden'); userProfile.classList.add('flex'); }
    else { loginBtn.classList.remove('hidden'); userProfile.classList.add('hidden'); userProfile.classList.remove('flex'); }
}

// --- NAVIGATION ---
function switchTab(activeTab) {
    tabs.forEach(tab => {
        const section = document.getElementById(`${tab}-section`);
        const btn = document.getElementById(`btn-${tab}`);
        if(section) section.classList.add('hidden');
        if(btn) btn.classList.remove('active');
    });
    
    const activeSection = document.getElementById(`${activeTab}-section`);
    if(activeSection) activeSection.classList.remove('hidden');
    
    const activeBtn = document.getElementById(`btn-${activeTab}`);
    if(activeBtn) activeBtn.classList.add('active');
    
    if(activeTab === 'dashboard') loadDashboard();
    if(activeTab === 'news') loadNews();
}

// --- DATA STORAGE (LOCAL STORAGE) ---
function getHistory() {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
}

function saveToHistory(type, details, outcome) {
    const history = getHistory();
    const entry = {
        date: new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }),
        type: type,
        details: details,
        outcome: outcome
    };
    
    history.unshift(entry);
    if (history.length > 10) history.pop();
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

// --- NEWS LOADER ---
async function loadNews() {
    const grid = document.getElementById('news-grid');
    if(!grid) return;
    grid.innerHTML = `<div class="col-span-full text-center py-20 text-slate-400"><i class="fa-solid fa-circle-notch fa-spin text-3xl mb-3 text-[#b4912f]"></i><p>Fetching legal briefs...</p></div>`;
    try {
        const res = await fetch('/api/news');
        const articles = await res.json();
        grid.innerHTML = '';
        articles.forEach(article => {
            const summary = article.summary.replace(/<[^>]*>?/gm, '').substring(0, 100) + '...';
            grid.innerHTML += `
                <a href="${article.link}" target="_blank" class="group bg-white border border-slate-200 rounded-lg overflow-hidden hover:shadow-xl transition-all duration-300 flex flex-col h-full hover:-translate-y-1">
                    <div class="h-40 overflow-hidden relative bg-slate-100">
                        <img src="${article.image}" alt="News" class="w-full h-full object-cover transition duration-700 group-hover:scale-105 opacity-90 group-hover:opacity-100">
                        <div class="absolute top-3 right-3 bg-white/90 backdrop-blur text-[#0f172a] text-[10px] font-bold px-2 py-1 shadow-sm uppercase tracking-wide">${article.source}</div>
                    </div>
                    <div class="p-6 flex flex-col flex-grow">
                        <h4 class="text-[#0f172a] font-serif font-bold text-lg mb-2 group-hover:text-[#b4912f] transition-colors leading-tight">${article.title}</h4>
                        <p class="text-slate-500 text-sm mb-4 flex-grow font-light leading-relaxed">${summary}</p>
                        <div class="flex items-center justify-between text-xs text-slate-400 mt-auto pt-4 border-t border-slate-100">
                            <span><i class="fa-regular fa-clock mr-1"></i> ${article.time}</span>
                            <span class="text-[#b4912f] font-semibold group-hover:underline">Full Story &rarr;</span>
                        </div>
                    </div>
                </a>`;
        });
    } catch(err) {
        grid.innerHTML = `<p class="col-span-full text-center text-slate-400">News feed unavailable.</p>`;
    }
}

// --- DASHBOARD ANALYTICS ---
async function loadDashboard() {
    const history = getHistory();
    const tbody = document.getElementById('history-table-body');
    
    if (history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="px-6 py-10 text-center text-slate-400 italic">No case history available.</td></tr>';
    } else {
        tbody.innerHTML = history.map(item => `
            <tr class="border-b border-slate-100 hover:bg-slate-50 transition">
                <td class="px-6 py-4 font-medium text-slate-600 whitespace-nowrap text-xs">${item.date}</td>
                <td class="px-6 py-4"><span class="bg-[#0f172a] text-white text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wide">${item.type}</span></td>
                <td class="px-6 py-4 text-slate-600 text-xs truncate max-w-xs" title="${item.details}">${item.details}</td>
                <td class="px-6 py-4 font-bold text-[#0f172a] text-sm">${item.outcome}</td>
            </tr>`).join('');
    }

    const counts = { 'Granted': 0, 'Dismissed': 0, 'Not Granted': 0, 'Withdrawn': 0 };

    history.forEach(h => {
        const out = h.outcome.toLowerCase();
        if (out.includes('grant') || out.includes('allow') || out.includes('award')) counts['Granted']++;
        else if (out.includes('dismiss')) counts['Dismissed']++;
        else if (out.includes('withdraw')) counts['Withdrawn']++;
        else if (out.includes('reject') || out.includes('denied')) counts['Not Granted']++;
        else counts['Granted']++;
    });

    const outcomeData = [
        counts['Granted'], 
        counts['Dismissed'], 
        counts['Not Granted'], 
        counts['Withdrawn']
    ];
    
    renderFeatureChart([85, 65, 55, 40, 25]); 
    renderOutcomeChart(outcomeData);
}

let featureChart, outcomeChart;


// --- 🔧 FIXED FEATURE CHART (Perfect Fit Sizing) ---
function renderFeatureChart(dataValues) {
    const canvas = document.getElementById('featureChart');
    if (!canvas) return;

    canvas.parentElement.style.height = "260px"; // Ensures perfect fit
    
    const ctx = canvas.getContext('2d');
    if (featureChart) featureChart.destroy();
    
    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Income Gap', 'Marriage Yrs', 'Evidence', 'Children', 'Assets'],
            datasets: [{ 
                label: 'Legal Weight (%)', 
                data: dataValues, 
                backgroundColor: '#0f172a', 
                borderRadius: 4,
                barThickness: 28,
                maxBarThickness: 35
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            plugins: { legend: { display: false } }, 
            scales: { 
                y: { beginAtZero: true, max: 100, grid: { color: '#e2e8f0' } }, 
                x: { grid: { display: false } }
            } 
        }
    });
}


// --- OUTCOME CHART ---
function renderOutcomeChart(dataValues) {
    const ctx = document.getElementById('outcomeChart').getContext('2d');
    if (outcomeChart) outcomeChart.destroy();
    
    const total = dataValues.reduce((a,b)=>a+b,0);
    const finalData = total === 0 ? [1,1,1,1] : dataValues;
    const bgColors = total === 0 
        ? ['#e2e8f0','#e2e8f0','#e2e8f0','#e2e8f0'] 
        : ['#0f172a', '#b4912f', '#334155', '#94a3b8'];

    outcomeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Granted', 'Dismissed', 'Not Granted', 'Withdrawn'],
            datasets: [{ 
                data: finalData, 
                backgroundColor: bgColors, 
                borderWidth: 0 
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            cutout: '70%', 
            plugins: { legend: { position: 'right', labels: { font: { family: "'Inter', sans-serif", size: 11 }, boxWidth: 12 } } } 
        }
    });
}

// --- FORM HANDLERS ---
function toggleWifeIncome() {
    const emp = document.getElementById('ali_w_employed').value;
    const inp = document.getElementById('ali_w');
    if(emp === "0") { inp.value=0; inp.disabled=true; inp.classList.add('bg-slate-100', 'text-slate-400'); }
    else { inp.disabled=false; inp.classList.remove('bg-slate-100', 'text-slate-400'); }
}

document.getElementById('alimonyForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    const txt = btn.innerText;
    btn.innerText = "Calculating..."; btn.disabled = true;
    
    const data = {
        husband_income: parseFloat(document.getElementById('ali_h').value)||0,
        wife_income: parseFloat(document.getElementById('ali_w').value)||0,
        is_wife_employed: parseInt(document.getElementById('ali_w_employed').value),
        marriage_duration: parseFloat(document.getElementById('ali_y').value)||0,
        children_count: parseFloat(document.getElementById('ali_k').value)||0,
        total_assets: parseFloat(document.getElementById('ali_a').value)||0
    };
    
    try {
        const res = await fetch('/predict/alimony', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
        if(!res.ok) throw new Error("Server Error");
        
        const r = await res.json();
        const resultStr = `₹${r.prediction.toLocaleString('en-IN')}`;
        
        saveToHistory('Alimony', `Assets: ₹${data.total_assets.toLocaleString()}`, resultStr);
        loadDashboard();

        const box = document.getElementById('ali-res');
        box.innerHTML = `
            <div class="text-center border-l-4 border-[#b4912f] bg-[#f8fafc] p-4 rounded-r">
                <p class="text-xs text-slate-500 font-bold uppercase tracking-widest mb-1">Estimated Settlement</p>
                <p class="text-4xl font-serif font-bold text-[#0f172a]">${resultStr}</p>
            </div>`;
        box.classList.remove('hidden');
    } catch(err) {
        alert("Calculation failed. Ensure server is running.");
    } finally { 
        btn.innerText = txt; 
        btn.disabled = false; 
    }
});

async function handleLegalSubmit(e, type) {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    const txt = btn.innerText;
    btn.innerText = "Analyzing..."; btn.disabled = true;
    
    const fd = new FormData(e.target);
    const d = Object.fromEntries(fd.entries());
    
    if(d.court_no) d.court_no = parseInt(d.court_no);
    if(d.year) d.year = parseInt(d.year);
    if(d.female_petitioner) d.female_petitioner = parseInt(d.female_petitioner);
    if(d.female_defendant) d.female_defendant = parseInt(d.female_defendant);
    
    try {
        const res = await fetch(`/predict/${type}`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(d) });
        if(!res.ok) throw new Error("Server Error");
        
        const r = await res.json();
        const box = document.getElementById(`${type}-res`);
        
        const catName = type.charAt(0).toUpperCase() + type.slice(1);
        saveToHistory(type === 'other' ? 'Disputes' : catName, d.dist_name || 'Court Case', r.outcome);
        
        loadDashboard();

        box.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <span class="font-bold text-slate-400 text-xs uppercase tracking-wider">AI Verdict</span>
                <span class="bg-[#0f172a] text-white text-[10px] font-bold px-2 py-0.5 rounded">CONFIDENCE: ${r.confidence}%</span>
            </div>
            <div class="text-center py-4 border-t border-slate-200 mt-2">
                <span class="text-2xl font-serif font-bold text-[#0f172a] block mb-2">${r.outcome}</span>
                <div class="w-full bg-slate-200 rounded-full h-1.5 mt-3 overflow-hidden">
                    <div class="bg-[#b4912f] h-1.5 rounded-full" style="width: ${r.confidence}%"></div>
                </div>
                <p class="text-[10px] text-right mt-1 text-slate-500">Probability: ${r.confidence}%</p>
            </div>`;
        box.classList.remove('hidden');
    } catch(err) {
        alert("Analysis failed. Ensure server is running.");
    } finally { 
        btn.innerText = txt; 
        btn.disabled = false; 
    }
}
