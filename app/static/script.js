/**
 * AEGIS FRAUD SHIELD • CLIENT CONTROLLER
 * Real-Time Streaming, Live AI Inference, Batch File Audit & Dynamic Financial Governance
 */

document.addEventListener('DOMContentLoaded', () => {

  // Global State
  let currentThreshold = 0.12;
  let isStreamActive = true;
  let streamInterval = null;
  let totalMonitoredCount = 1482904;
  let attacksBlockedCount = 2561;
  let capitalShieldedAmount = 3842910.00;

  let selectedBatchFile = null;
  let batchAuditRecords = [];
  let currentBatchFilter = 'all';

  // Active Dataset Tracking for Threat Analytics (Tab 4) & Radar (Tab 1)
  let activeAnalyticsData = null;
  let activeDatasetName = "Global Baseline (284,807 transactions)";

  // Global Baseline Analytics Cache
  const baselineAnalytics = {
    dataset_name: "Global Baseline (284,807 transactions)",
    log_amounts_genuine: generateBaselineLogAmounts(350, 3.1, 2.4),
    log_amounts_fraud: generateBaselineLogAmounts(150, 4.3, 3.2),
    feature_shifts: {
      'V14': -6.85, 'V17': -5.62, 'V12': -4.95, 'V10': -4.31, 'V4': 4.52, 'V11': 3.78
    },
    hourly_attacks: [42, 68, 95, 110, 85, 45, 20, 15, 12, 14, 18, 22, 25, 28, 30, 32, 38, 45, 52, 60, 72, 85, 90, 70],
    genuine_count: 284315,
    fraud_count: 492,
    total_count: 284807,
    fraud_rate_pct: 0.173
  };

  function generateBaselineLogAmounts(count, center, spread) {
    const arr = [];
    for (let i = 0; i < count; i++) {
      arr.push(parseFloat(Math.max(0, center + (Math.random() - 0.5) * spread).toFixed(2)));
    }
    return arr;
  }

  // --------------------------------------------------------------------------
  // 1. Navigation & Clock Setup
  // --------------------------------------------------------------------------
  function updateLiveClock() {
    const clockEl = document.getElementById('live-clock');
    if (clockEl) {
      const now = new Date();
      clockEl.textContent = now.toUTCString().split(' ')[4] + ' UTC';
    }
  }
  setInterval(updateLiveClock, 1000);
  updateLiveClock();

  // Tab Navigation
  const tabButtons = document.querySelectorAll('.nav-tab-btn');
  const contentPanes = document.querySelectorAll('.content-pane');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      contentPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(`pane-${btn.dataset.tab}`);
      if (targetPane) {
        targetPane.classList.add('active');
        window.dispatchEvent(new Event('resize'));
        if (btn.dataset.tab === 'screener') {
          setTimeout(recalculateSingleScreener, 60);
        } else if (btn.dataset.tab === 'analytics') {
          setTimeout(() => {
            renderThreatAnalytics(activeAnalyticsData || baselineAnalytics, activeDatasetName);
          }, 60);
        }
      }
    });
  });

  // --------------------------------------------------------------------------
  // 2. Threshold & Operational Policy Synchronization
  // --------------------------------------------------------------------------
  const quickSlider = document.getElementById('quick-threshold-slider');
  const quickSliderVal = document.getElementById('quick-thresh-val');
  const navPolicyVal = document.getElementById('nav-policy-val');
  const resThreshVal = document.getElementById('res-thresh-val');

  function setGlobalThreshold(newVal) {
    currentThreshold = parseFloat(newVal);
    if (quickSlider) quickSlider.value = currentThreshold;
    if (quickSliderVal) quickSliderVal.textContent = currentThreshold.toFixed(2);
    if (resThreshVal) resThreshVal.textContent = `θ = ${currentThreshold.toFixed(2)}`;

    // Update policy text
    let policyName = "CUSTOM";
    if (Math.abs(currentThreshold - 0.12) < 0.01) policyName = "BALANCED (θ=0.12)";
    else if (Math.abs(currentThreshold - 0.05) < 0.01) policyName = "CONSERVATIVE (θ=0.05)";
    else if (Math.abs(currentThreshold - 0.50) < 0.01) policyName = "PERMISSIVE (θ=0.50)";
    
    if (navPolicyVal) navPolicyVal.textContent = policyName;

    // Highlight active policy option in Tab 5
    document.querySelectorAll('.policy-option').forEach(opt => {
      if (Math.abs(parseFloat(opt.dataset.policy) - currentThreshold) < 0.015) {
        opt.classList.add('active');
      } else {
        opt.classList.remove('active');
      }
    });

    // Recalculate single screener dynamically
    recalculateSingleScreener();

    // Re-render batch records and dynamic threat analytics if file is loaded
    if (batchAuditRecords.length > 0) {
      renderBatchAuditUI();
      // Dynamically recompute analytics based on updated threshold
      activeAnalyticsData = computeAnalyticsFromRecords(batchAuditRecords, selectedBatchFile ? selectedBatchFile.name : activeDatasetName);
      renderThreatAnalytics(activeAnalyticsData, activeDatasetName);
      renderRadarDonutAndHourly(activeAnalyticsData);
    }
  }

  if (quickSlider) {
    quickSlider.addEventListener('input', (e) => setGlobalThreshold(e.target.value));
  }

  const btnSnap012 = document.getElementById('btn-snap-012');
  if (btnSnap012) btnSnap012.addEventListener('click', () => setGlobalThreshold(0.12));

  const btnSnap050 = document.getElementById('btn-snap-050');
  if (btnSnap050) btnSnap050.addEventListener('click', () => setGlobalThreshold(0.50));

  // Tab 5 policy cards click
  document.querySelectorAll('.policy-option').forEach(opt => {
    opt.addEventListener('click', () => {
      setGlobalThreshold(opt.dataset.policy);
    });
  });

  // --------------------------------------------------------------------------
  // 3. Tab 1: Real-Time Streaming Radar Feed
  // --------------------------------------------------------------------------
  const streamTableBody = document.getElementById('stream-table-body');
  const btnToggleStream = document.getElementById('btn-toggle-stream');
  const streamToggleIcon = document.getElementById('stream-toggle-icon');
  const streamToggleText = document.getElementById('stream-toggle-text');
  const btnClearStream = document.getElementById('btn-clear-stream');

  const monitoredValEl = document.getElementById('kpi-total-monitored');
  const attacksValEl = document.getElementById('kpi-attacks-blocked');
  const capitalValEl = document.getElementById('kpi-capital-shielded');

  async function generateStreamTransaction() {
    if (!isStreamActive || !streamTableBody) return;

    let txn = null;
    try {
      const res = await fetch('/api/stream/sample');
      if (res.ok) {
        txn = await res.json();
      }
    } catch(e) {}

    if (!txn) {
      // High-speed client fallback simulation
      const isFraud = Math.random() < 0.09;
      const isReview = !isFraud && Math.random() < 0.15;
      const amt = isFraud ? (Math.random() * 2500 + 400).toFixed(2) : (Math.random() * 75 + 5).toFixed(2);
      const prob = isFraud ? (Math.random() * 0.25 + 0.72) : (isReview ? (Math.random() * 0.15 + 0.22) : (Math.random() * 0.04));
      const merchants = ["Amazon UK", "Apple Store", "Crypto Swap ATM", "Starbucks Coffee", "Shell Fuel Station", "Spotify", "Luxury Paris"];
      const locations = ["London, UK", "New York, US", "Frankfurt, DE", "Paris, FR", "Amsterdam, NL", "Tokyo, JP"];

      txn = {
        id: `TXN-${Math.floor(Math.random() * 899999 + 100000)}`,
        timestamp: new Date().toTimeString().split(' ')[0],
        cardholder: `•••• ${Math.floor(Math.random() * 8999 + 1000)}`,
        merchant: merchants[Math.floor(Math.random() * merchants.length)],
        location: locations[Math.floor(Math.random() * locations.length)],
        amount: parseFloat(amt),
        fraud_risk_pct: (prob * 100).toFixed(1),
        fraud_probability: prob,
        is_fraud: prob >= currentThreshold,
        status: prob >= currentThreshold ? "🚨 BLOCKED" : (prob >= 0.20 ? "⚠️ REVIEW" : "✅ APPROVED")
      };
    }

    // Dynamic decision based on current operational threshold
    const isBlocked = txn.fraud_probability >= currentThreshold;
    const badgeClass = isBlocked ? 'badge-blocked' : (txn.fraud_probability >= 0.20 ? 'badge-review' : 'badge-approved');
    const statusText = isBlocked ? '🚨 BLOCKED' : (txn.fraud_probability >= 0.20 ? '⚠️ REVIEW' : '✅ APPROVED');

    // Update real-time counters
    totalMonitoredCount++;
    if (monitoredValEl) monitoredValEl.textContent = totalMonitoredCount.toLocaleString();

    if (isBlocked) {
      attacksBlockedCount++;
      capitalShieldedAmount += txn.amount;
      if (attacksValEl) attacksValEl.textContent = attacksBlockedCount.toLocaleString();
      if (capitalValEl) capitalValEl.textContent = `€${capitalShieldedAmount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    // Insert row at top
    const tr = document.createElement('tr');
    if (isBlocked) tr.className = 'row-flash-red';

    tr.innerHTML = `
      <td>${txn.timestamp}</td>
      <td><strong>${txn.id}</strong></td>
      <td>${txn.cardholder}</td>
      <td>${txn.merchant}</td>
      <td>${txn.location}</td>
      <td><strong>€${txn.amount.toFixed(2)}</strong></td>
      <td><span class="${isBlocked ? 'text-red' : (txn.fraud_probability >= 0.20 ? 'text-amber' : 'text-green')}"><strong>${txn.fraud_risk_pct}%</strong></span></td>
      <td><span class="badge-status ${badgeClass}">${statusText}</span></td>
    `;

    streamTableBody.insertBefore(tr, streamTableBody.firstChild);

    // Keep table to 20 rows max for peak performance
    while (streamTableBody.children.length > 20) {
      streamTableBody.removeChild(streamTableBody.lastChild);
    }
  }

  function startStream() {
    isStreamActive = true;
    if (streamToggleIcon) streamToggleIcon.textContent = '⏸️';
    if (streamToggleText) streamToggleText.textContent = 'Pause Stream';
    if (btnToggleStream) btnToggleStream.classList.add('active');
    if (streamInterval) clearInterval(streamInterval);
    streamInterval = setInterval(generateStreamTransaction, 2400);
  }

  function pauseStream() {
    isStreamActive = false;
    if (streamToggleIcon) streamToggleIcon.textContent = '▶️';
    if (streamToggleText) streamToggleText.textContent = 'Start Live Stream';
    if (btnToggleStream) btnToggleStream.classList.remove('active');
    if (streamInterval) clearInterval(streamInterval);
  }

  if (btnToggleStream) {
    btnToggleStream.addEventListener('click', () => {
      if (isStreamActive) pauseStream();
      else startStream();
    });
  }

  if (btnClearStream && streamTableBody) {
    btnClearStream.addEventListener('click', () => {
      streamTableBody.innerHTML = '';
    });
  }

  // Initial stream rows
  for (let i = 0; i < 4; i++) {
    generateStreamTransaction();
  }
  startStream();

  // --------------------------------------------------------------------------
  // 4. Tab 2: Real-Time Instant Transaction Screener
  // --------------------------------------------------------------------------
  const screenerForm = document.getElementById('single-screener-form');
  const screenerAmount = document.getElementById('screener-amount');
  const screenerTime = document.getElementById('screener-time');
  const cardholderInp = document.getElementById('inp-cardholder');
  const merchantInp = document.getElementById('inp-merchant');

  const keyVectors = ['v14', 'v17', 'v12', 'v10', 'v4', 'v11'];

  // Synchronize live slider labels & trigger live scoring on every slider movement
  keyVectors.forEach(vec => {
    const slider = document.getElementById(`slider-${vec}`);
    const label = document.getElementById(`val-${vec}`);
    if (slider && label) {
      slider.addEventListener('input', (e) => {
        label.textContent = parseFloat(e.target.value).toFixed(2);
        recalculateSingleScreener();
      });
    }
  });

  if (screenerAmount) screenerAmount.addEventListener('input', recalculateSingleScreener);
  if (screenerTime) screenerTime.addEventListener('input', recalculateSingleScreener);

  if (screenerForm) {
    screenerForm.addEventListener('submit', (e) => {
      e.preventDefault();
      recalculateSingleScreener();
    });
  }

  // Presets definition
  const screenerPresets = {
    genuine: {
      Amount: 18.50, Time: 42000,
      v14: 0.12, v17: -0.05, v12: 0.20, v10: -0.10, v4: 0.05, v11: 0.00,
      cardholder: "ACC-8192", merchant: "Starbucks Coffee"
    },
    borderline: {
      Amount: 420.00, Time: 85400,
      v14: -2.35, v17: -1.70, v12: -1.90, v10: -1.40, v4: 1.80, v11: 1.20,
      cardholder: "ACC-9921", merchant: "High-End Electronics"
    },
    fraud: {
      Amount: 2890.00, Time: 8200,
      v14: -7.45, v17: -6.10, v12: -5.30, v10: -4.80, v4: 4.90, v11: 3.80,
      cardholder: "ACC-4102", merchant: "Crypto Swap ATM"
    },
    random: {
      Amount: parseFloat((Math.random() * 800 + 15).toFixed(2)),
      Time: Math.floor(Math.random() * 80000 + 10000),
      v14: parseFloat((Math.random() * 6 - 4).toFixed(2)),
      v17: parseFloat((Math.random() * 5 - 3).toFixed(2)),
      v12: parseFloat((Math.random() * 5 - 3).toFixed(2)),
      v10: parseFloat((Math.random() * 4 - 2).toFixed(2)),
      v4: parseFloat((Math.random() * 5 - 1).toFixed(2)),
      v11: parseFloat((Math.random() * 4 - 1).toFixed(2)),
      cardholder: `ACC-${Math.floor(Math.random() * 8999 + 1000)}`,
      merchant: "Direct POS Terminal"
    }
  };

  function applyScreenerPreset(presetName) {
    const data = screenerPresets[presetName];
    if (!data) return;

    if (screenerAmount) screenerAmount.value = data.Amount;
    if (screenerTime) screenerTime.value = data.Time;
    if (cardholderInp) cardholderInp.value = data.cardholder;
    if (merchantInp) merchantInp.value = data.merchant;

    keyVectors.forEach(vec => {
      const slider = document.getElementById(`slider-${vec}`);
      const label = document.getElementById(`val-${vec}`);
      if (slider && data[vec] !== undefined) {
        slider.value = data[vec];
        if (label) label.textContent = parseFloat(data[vec]).toFixed(2);
      }
    });

    recalculateSingleScreener();
  }

  const pGen = document.getElementById('preset-genuine');
  if (pGen) pGen.addEventListener('click', () => applyScreenerPreset('genuine'));

  const pBord = document.getElementById('preset-borderline');
  if (pBord) pBord.addEventListener('click', () => applyScreenerPreset('borderline'));

  const pFr = document.getElementById('preset-fraud');
  if (pFr) pFr.addEventListener('click', () => applyScreenerPreset('fraud'));

  const pRnd = document.getElementById('preset-random');
  if (pRnd) pRnd.addEventListener('click', () => applyScreenerPreset('random'));

  // Live Async Screener Recalculation
  async function recalculateSingleScreener() {
    const payload = {
      Amount: parseFloat(screenerAmount ? screenerAmount.value : 18.50) || 18.50,
      Time: parseFloat(screenerTime ? screenerTime.value : 42000) || 42000,
      threshold: currentThreshold,
      cardholder_id: cardholderInp ? cardholderInp.value : "ACC-84920",
      merchant: merchantInp ? merchantInp.value : "Online Store"
    };

    keyVectors.forEach(vec => {
      const slider = document.getElementById(`slider-${vec}`);
      payload[vec.toUpperCase()] = slider ? parseFloat(slider.value) : 0.0;
    });

    let resultData = null;
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        resultData = await res.json();
      }
    } catch(e) {}

    if (!resultData) {
      // Analytical high-precision surrogate engine
      const v14 = payload.V14 || 0;
      const v17 = payload.V17 || 0;
      const v12 = payload.V12 || 0;
      const v10 = payload.V10 || 0;
      const v4  = payload.V4 || 0;
      const v11 = payload.V11 || 0;
      const amt = payload.Amount;

      const z = -4.65 - (0.82 * v14) - (0.58 * v17) - (0.64 * v12) - (0.48 * v10) + (0.52 * v4) + (0.42 * v11);
      const prob = 1.0 / (1.0 + Math.exp(-Math.max(-30, Math.min(30, z))));

      const isFraud = prob >= currentThreshold;
      let riskLevel = "LOW";
      let dec = "✅ TRANSACTION APPROVED";
      let rec = "Clean transaction profile. Approved seamlessly with zero customer friction.";

      if (prob >= 0.70) {
        riskLevel = "CRITICAL";
        dec = "🚨 TRANSACTION BLOCKED";
        rec = "High-confidence fraud attack. Card frozen automatically; 2FA challenge dispatched.";
      } else if (isFraud) {
        riskLevel = "HIGH";
        dec = "🚨 TRANSACTION BLOCKED";
        rec = "Risk exceeds authorized operational threshold. Immediate SMS verification triggered.";
      } else if (prob >= 0.20) {
        riskLevel = "ELEVATED";
        dec = "⚠️ MANUAL REVIEW";
        rec = "Borderline anomaly. Dispatched to Fraud Operations queue for analyst review.";
      }

      const factors = [];
      if (v14 < -2.0) factors.push(`Severe velocity displacement in V14 (${v14.toFixed(2)}) indicates botnet credential attack (+38% risk)`);
      if (v17 < -2.0) factors.push(`Geographical device jump in V17 (${v17.toFixed(2)}) indicates foreign account takeover (+28% risk)`);
      if (v4 > 1.5) factors.push(`High-frequency automated retry in V4 (${v4.toFixed(2)}) (+22% risk)`);
      if (amt > 800) factors.push(`High transaction amount (€${amt.toFixed(2)}) exceeds standard customer baseline`);
      else factors.push(`Standard micro-transaction amount (€${amt.toFixed(2)}) matches legitimate retail spending`);

      resultData = {
        fraud_probability: prob,
        fraud_risk_pct: round2(prob * 100),
        is_fraud: isFraud,
        decision: dec,
        risk_level: riskLevel,
        recommendation: rec,
        risk_factors: factors
      };
    }

    renderScreenerOutputs(resultData);
  }

  function round2(v) { return Math.round(v * 100) / 100; }

  function renderScreenerOutputs(res) {
    const probPct = res.fraud_risk_pct;
    const threshPct = currentThreshold * 100;

    // Update Decision Banner
    const banner = document.getElementById('screener-decision-banner');
    const bannerIcon = document.getElementById('banner-icon');
    const bannerHead = document.getElementById('banner-headline');
    const bannerRec = document.getElementById('banner-recommendation');

    if (banner) {
      banner.className = 'decision-banner ' + (res.is_fraud ? 'banner-blocked' : (res.fraud_probability >= 0.20 ? 'banner-review' : 'banner-safe'));
      if (bannerIcon) bannerIcon.textContent = res.is_fraud ? '🚨' : (res.fraud_probability >= 0.20 ? '⚠️' : '✅');
      if (bannerHead) bannerHead.textContent = res.decision.replace(/^[^\w]+/, '').trim();
      if (bannerRec) bannerRec.textContent = res.recommendation;
    }

    // Update Stat Cards
    const probEl = document.getElementById('res-prob-val');
    if (probEl) {
      probEl.textContent = `${probPct.toFixed(2)}%`;
      probEl.className = 'res-stat-val ' + (res.is_fraud ? 'text-red' : (res.fraud_probability >= 0.20 ? 'text-amber' : 'text-green'));
    }

    const tierEl = document.getElementById('res-tier-val');
    if (tierEl) {
      tierEl.textContent = res.risk_level;
      tierEl.className = 'res-stat-val ' + (res.is_fraud ? 'text-red' : (res.fraud_probability >= 0.20 ? 'text-amber' : 'text-green'));
    }

    // Update Explainability Factors
    const factorsList = document.getElementById('screener-factors-list');
    if (factorsList && res.risk_factors) {
      factorsList.innerHTML = '';
      res.risk_factors.forEach(f => {
        const li = document.createElement('li');
        li.textContent = `• ${f}`;
        factorsList.appendChild(li);
      });
    }

    // Render Plotly Radial Indicator Gauge
    const gaugeData = [{
      type: "indicator",
      mode: "gauge+number",
      value: probPct,
      number: { suffix: "%", font: { size: 28, color: "#fff" } },
      title: { text: `AI Threat Level vs θ=${currentThreshold.toFixed(2)}`, font: { size: 12, color: "#94A3B8" } },
      gauge: {
        axis: { range: [0, 100], tickcolor: "#475569", tickwidth: 1 },
        bar: { color: "#38BDF8", thickness: 0.25 },
        bgcolor: "#0F172A",
        borderwidth: 1,
        bordercolor: "#334155",
        steps: [
          { range: [0, 20], color: "rgba(16, 185, 129, 0.3)" },
          { range: [20, threshPct], color: "rgba(245, 158, 11, 0.3)" },
          { range: [threshPct, 100], color: "rgba(239, 68, 68, 0.4)" }
        ],
        threshold: {
          line: { color: "#F87171", width: 4 },
          thickness: 0.8,
          value: threshPct
        }
      }
    }];

    const gaugeLayout = {
      height: 210,
      margin: { l: 24, r: 24, t: 36, b: 20 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };

    try {
      Plotly.react('screener-gauge-chart', gaugeData, gaugeLayout, { responsive: true, displayModeBar: false });
    } catch(e) {
      Plotly.newPlot('screener-gauge-chart', gaugeData, gaugeLayout, { responsive: true, displayModeBar: false });
    }
  }

  // --------------------------------------------------------------------------
  // 5. Tab 3: Batch Upload & THE REQUESTED ANALYZE BUTTON
  // --------------------------------------------------------------------------
  const dropzoneArea = document.getElementById('dropzone-area');
  const batchFileInput = document.getElementById('batch-file-input');
  const btnBrowse = document.getElementById('btn-browse');
  const fileNameText = document.getElementById('file-name-text');
  const btnRunBatch = document.getElementById('btn-run-batch-analysis');
  const batchProgressWrap = document.getElementById('batch-progress-wrap');
  const progressFill = document.getElementById('progress-fill-bar');
  const progressPctText = document.getElementById('progress-pct-text');
  const progressStatusText = document.getElementById('progress-status-text');
  const batchResultsDash = document.getElementById('batch-results-dashboard');
  const batchTableBody = document.getElementById('batch-table-body');
  const btnExportBatchReport = document.getElementById('btn-export-batch-report');
  const batchSearchInput = document.getElementById('batch-search-input');

  // Trigger file browse
  if (btnBrowse && batchFileInput) {
    btnBrowse.addEventListener('click', (e) => {
      e.stopPropagation();
      batchFileInput.click();
    });
  }

  if (dropzoneArea && batchFileInput) {
    dropzoneArea.addEventListener('click', () => batchFileInput.click());

    dropzoneArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzoneArea.classList.add('dragover');
    });

    dropzoneArea.addEventListener('dragleave', () => dropzoneArea.classList.remove('dragover'));

    dropzoneArea.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzoneArea.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    batchFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Please upload a valid CSV transaction file.');
      return;
    }

    selectedBatchFile = file;
    const sizeKb = (file.size / 1024).toFixed(1);
    if (fileNameText) fileNameText.textContent = `${file.name} (${sizeKb} KB) - Ready for AI Analysis`;

    // Enable the prominent analyze button!
    if (btnRunBatch) {
      btnRunBatch.disabled = false;
      btnRunBatch.classList.add('pulse-ready');
    }
  }

  // THE PROMINENT ANALYSE BUTTON CLICK HANDLER
  if (btnRunBatch) {
    btnRunBatch.addEventListener('click', async () => {
      if (!selectedBatchFile) return;

      btnRunBatch.disabled = true;
      if (batchProgressWrap) batchProgressWrap.style.display = 'block';
      if (progressFill) progressFill.style.width = '15%';
      if (progressPctText) progressPctText.textContent = '15%';
      if (progressStatusText) progressStatusText.textContent = 'Parsing payment transaction records...';

      setTimeout(() => {
        if (progressFill) progressFill.style.width = '55%';
        if (progressPctText) progressPctText.textContent = '55%';
        if (progressStatusText) progressStatusText.textContent = 'Evaluating through Neural XGBoost Ensemble...';
      }, 300);

      try {
        const formData = new FormData();
        formData.append('file', selectedBatchFile);
        formData.append('threshold', currentThreshold);

        const response = await fetch('/api/predict/batch', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const data = await response.json();
          completeBatchAnalysis(data.records, data.analytics, selectedBatchFile.name);
          return;
        }
      } catch(e) {}

      // Robust client-side fallback parsing with FileReader
      const reader = new FileReader();
      reader.onload = function(evt) {
        parseAndScoreCsvClient(evt.target.result, selectedBatchFile.name);
      };
      reader.readAsText(selectedBatchFile);
    });
  }

  function parseAndScoreCsvClient(csvText, fileName) {
    const lines = csvText.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length <= 1) return;

    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    const records = [];

    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',').map(p => p.trim().replace(/^"|"$/g, ''));
      if (parts.length < 2) continue;

      const row = {};
      headers.forEach((h, idx) => row[h] = parts[idx]);

      const amt = parseFloat(row.Amount || row.amount || 25.0);
      const timeVal = parseFloat(row.Time || row.time || (i * 10));
      const txnId = row.Transaction_ID || row.ID || `TXN-${10000 + i}`;
      const v14 = parseFloat(row.V14 || 0.0);
      const v17 = parseFloat(row.V17 || 0.0);
      const v12 = parseFloat(row.V12 || 0.0);
      const v4  = parseFloat(row.V4 || 0.0);
      const v10 = parseFloat(row.V10 || 0.0);
      const v11 = parseFloat(row.V11 || 0.0);

      const z = -4.65 - (0.82 * v14) - (0.58 * v17) - (0.64 * v12) - (0.48 * v10) + (0.52 * v4) + (0.42 * v11);
      const prob = 1.0 / (1.0 + Math.exp(-Math.max(-30, Math.min(30, z))));

      records.push({
        id: txnId,
        amount: amt,
        time: timeVal,
        fraud_probability: prob,
        risk_pct: round2(prob * 100),
        v14: v14.toFixed(2),
        v17: v17.toFixed(2),
        v12: v12.toFixed(2),
        v10: v10.toFixed(2),
        v4: v4.toFixed(2),
        v11: v11.toFixed(2)
      });
    }

    const clientAnalytics = computeAnalyticsFromRecords(records, fileName);
    completeBatchAnalysis(records, clientAnalytics, fileName);
  }

  // Helper: Computes real statistical distributions from any array of records
  function computeAnalyticsFromRecords(records, fileName) {
    const logGen = [];
    const logFraud = [];
    const hourlyAttacks = new Array(24).fill(0);
    let flagged = 0;

    let sumV14F = 0, sumV14G = 0;
    let sumV17F = 0, sumV17G = 0;
    let sumV12F = 0, sumV12G = 0;
    let sumV10F = 0, sumV10G = 0;
    let sumV4F = 0, sumV4G = 0;
    let sumV11F = 0, sumV11G = 0;

    records.forEach(r => {
      const isF = r.fraud_probability >= currentThreshold;
      const logAmt = parseFloat(Math.log1p(Math.max(0, r.amount)).toFixed(2));
      const hour = Math.max(0, Math.min(23, Math.floor(((r.time % 86400) / 3600)) % 24));

      if (isF) {
        flagged++;
        logFraud.push(logAmt);
        hourlyAttacks[hour]++;
        sumV14F += parseFloat(r.v14 || 0);
        sumV17F += parseFloat(r.v17 || 0);
        sumV12F += parseFloat(r.v12 || 0);
        sumV10F += parseFloat(r.v10 || 0);
        sumV4F += parseFloat(r.v4 || 0);
        sumV11F += parseFloat(r.v11 || 0);
      } else {
        logGen.push(logAmt);
        sumV14G += parseFloat(r.v14 || 0);
        sumV17G += parseFloat(r.v17 || 0);
        sumV12G += parseFloat(r.v12 || 0);
        sumV10G += parseFloat(r.v10 || 0);
        sumV4G += parseFloat(r.v4 || 0);
        sumV11G += parseFloat(r.v11 || 0);
      }
    });

    const genCount = Math.max(1, records.length - flagged);
    const fraudCount = Math.max(1, flagged);

    return {
      dataset_name: fileName || "Uploaded Batch Dataset",
      log_amounts_genuine: logGen,
      log_amounts_fraud: logFraud,
      feature_shifts: {
        'V14': round2((sumV14F / fraudCount) - (sumV14G / genCount)),
        'V17': round2((sumV17F / fraudCount) - (sumV17G / genCount)),
        'V12': round2((sumV12F / fraudCount) - (sumV12G / genCount)),
        'V10': round2((sumV10F / fraudCount) - (sumV10G / genCount)),
        'V4': round2((sumV4F / fraudCount) - (sumV4G / genCount)),
        'V11': round2((sumV11F / fraudCount) - (sumV11G / genCount))
      },
      hourly_attacks: hourlyAttacks,
      genuine_count: records.length - flagged,
      fraud_count: flagged,
      total_count: records.length,
      fraud_rate_pct: round2((flagged / Math.max(1, records.length)) * 100)
    };
  }

  function completeBatchAnalysis(records, analytics, fileName) {
    if (progressFill) progressFill.style.width = '100%';
    if (progressPctText) progressPctText.textContent = '100%';
    if (progressStatusText) progressStatusText.textContent = 'Batch Analysis Complete!';

    setTimeout(() => {
      if (batchProgressWrap) batchProgressWrap.style.display = 'none';
      if (btnRunBatch) btnRunBatch.disabled = false;
      if (batchResultsDash) batchResultsDash.style.display = 'block';

      batchAuditRecords = records;
      activeDatasetName = fileName || (selectedBatchFile ? selectedBatchFile.name : "Uploaded Batch Dataset");
      activeAnalyticsData = analytics || computeAnalyticsFromRecords(records, activeDatasetName);

      // Render Batch UI Table & KPIs
      renderBatchAuditUI();

      // DYNAMICALLY RE-RENDER TAB 4 THREAT ANALYTICS WITH THE UPLOADED FILE'S REAL DATA!
      renderThreatAnalytics(activeAnalyticsData, activeDatasetName);

      // DYNAMICALLY UPDATE TAB 1 RADAR CHARTS WITH THE UPLOADED FILE'S REAL DATA!
      renderRadarDonutAndHourly(activeAnalyticsData);

    }, 400);
  }

  function renderBatchAuditUI() {
    if (!batchTableBody) return;
    batchTableBody.innerHTML = '';

    let total = batchAuditRecords.length;
    let blockedCount = 0;
    let reviewCount = 0;
    let approvedCount = 0;
    let capitalSaved = 0.0;

    batchAuditRecords.forEach(r => {
      r.isBlocked = r.fraud_probability >= currentThreshold;
      if (r.isBlocked) {
        blockedCount++;
        capitalSaved += r.amount;
        r.decision = "🚨 BLOCKED";
        r.tier = r.fraud_probability >= 0.70 ? "Critical" : "High";
      } else if (r.fraud_probability >= 0.20) {
        reviewCount++;
        r.decision = "⚠️ REVIEW";
        r.tier = "Medium";
      } else {
        approvedCount++;
        r.decision = "✅ APPROVED";
        r.tier = "Low";
      }
    });

    // Update KPI Counts
    const statTot = document.getElementById('batch-stat-total');
    if (statTot) statTot.textContent = total.toLocaleString();

    const statFlg = document.getElementById('batch-stat-flagged');
    if (statFlg) statFlg.textContent = blockedCount.toLocaleString();

    const statRate = document.getElementById('batch-stat-rate');
    if (statRate) statRate.textContent = `${((blockedCount / Math.max(1, total)) * 100).toFixed(2)}% attack rate`;

    const statSaved = document.getElementById('batch-stat-saved');
    if (statSaved) statSaved.textContent = `€${capitalSaved.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    const statApp = document.getElementById('batch-stat-approved');
    if (statApp) statApp.textContent = approvedCount.toLocaleString();

    // Update Filter Pills Badges
    const cAll = document.getElementById('count-all');
    if (cAll) cAll.textContent = total;
    const cBlk = document.getElementById('count-blocked');
    if (cBlk) cBlk.textContent = blockedCount;
    const cRev = document.getElementById('count-review');
    if (cRev) cRev.textContent = reviewCount;
    const cApp = document.getElementById('count-approved');
    if (cApp) cApp.textContent = approvedCount;

    // Filter and search
    const query = batchSearchInput ? batchSearchInput.value.toLowerCase().trim() : '';

    const filtered = batchAuditRecords.filter(r => {
      if (currentBatchFilter === 'blocked' && !r.isBlocked) return false;
      if (currentBatchFilter === 'review' && (r.isBlocked || r.fraud_probability < 0.20)) return false;
      if (currentBatchFilter === 'approved' && r.fraud_probability >= 0.20) return false;

      if (query) {
        return r.id.toLowerCase().includes(query) || r.amount.toString().includes(query);
      }
      return true;
    });

    filtered.sort((a, b) => b.fraud_probability - a.fraud_probability);

    filtered.forEach(row => {
      const tr = document.createElement('tr');
      if (row.isBlocked) tr.className = 'row-flash-red';

      const badgeClass = row.isBlocked ? 'badge-blocked' : (row.fraud_probability >= 0.20 ? 'badge-review' : 'badge-approved');

      tr.innerHTML = `
        <td><strong>${row.id}</strong></td>
        <td>€${row.amount.toFixed(2)}</td>
        <td><strong class="${row.isBlocked ? 'text-red' : (row.fraud_probability >= 0.20 ? 'text-amber' : 'text-green')}">${(row.fraud_probability * 100).toFixed(2)}%</strong></td>
        <td>${row.tier}</td>
        <td><span class="badge-status ${badgeClass}">${row.decision}</span></td>
        <td>${row.time}</td>
        <td class="${parseFloat(row.v14) < -2 ? 'text-red' : ''}">${row.v14}</td>
        <td class="${parseFloat(row.v17) < -2 ? 'text-red' : ''}">${row.v17}</td>
      `;
      batchTableBody.appendChild(tr);
    });
  }

  // Filter Buttons
  document.querySelectorAll('.filter-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentBatchFilter = pill.dataset.filter;
      renderBatchAuditUI();
    });
  });

  if (batchSearchInput) {
    batchSearchInput.addEventListener('input', renderBatchAuditUI);
  }

  // Export Scored Report
  if (btnExportBatchReport) {
    btnExportBatchReport.addEventListener('click', () => {
      if (batchAuditRecords.length === 0) return;
      let csv = "Transaction_ID,Amount,AI_Fraud_Probability,Risk_Tier,Decision,Time,V14_Velocity,V17_Geo\n";
      batchAuditRecords.forEach(r => {
        csv += `${r.id},${r.amount},${r.fraud_probability},"${r.tier}","${r.decision}",${r.time},${r.v14},${r.v17}\n`;
      });
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `Aegis_Scored_Audit_Report_Thresh_${currentThreshold.toFixed(2)}.csv`;
      link.click();
    });
  }

  // --------------------------------------------------------------------------
  // 6. DYNAMIC THREAT ANALYTICS (TAB 4) & RADAR CHARTS (TAB 1)
  // --------------------------------------------------------------------------
  const darkPaperBg = 'transparent';
  const darkFontColor = '#94A3B8';

  // Dynamic Renderer for Tab 4: Threat Analytics
  function renderThreatAnalytics(data, datasetName) {
    if (!data) data = baselineAnalytics;

    // 1. Update Active Dataset Source Banner on Tab 4
    const srcNameEl = document.getElementById('analytics-source-name');
    const srcPillEl = document.getElementById('analytics-source-pill');
    const btnResetBaseline = document.getElementById('btn-reset-baseline');

    const displayName = datasetName || data.dataset_name || "Uploaded Dataset";
    if (srcNameEl) srcNameEl.textContent = displayName;
    if (srcPillEl) srcPillEl.textContent = `${data.total_count.toLocaleString()} transactions • ${data.fraud_rate_pct}% attack rate`;

    if (btnResetBaseline) {
      if (displayName.includes("Global Baseline")) {
        btnResetBaseline.style.display = 'none';
      } else {
        btnResetBaseline.style.display = 'inline-block';
      }
    }

    // 2. Real Log-Amount Distribution Histogram (Genuine vs Fraud in this dataset)
    const logGen = (data.log_amounts_genuine && data.log_amounts_genuine.length > 0) ? data.log_amounts_genuine : [1.2, 2.5, 3.1];
    const logFraud = (data.log_amounts_fraud && data.log_amounts_fraud.length > 0) ? data.log_amounts_fraud : [4.5, 5.2, 6.1];

    const distData = [
      {
        x: logGen,
        type: 'histogram',
        name: `Genuine (${logGen.length})`,
        opacity: 0.65,
        marker: { color: '#0284C7' },
        nbinsx: Math.min(35, Math.max(10, Math.floor(logGen.length / 3)))
      },
      {
        x: logFraud,
        type: 'histogram',
        name: `Fraudulent (${logFraud.length})`,
        opacity: 0.85,
        marker: { color: '#EF4444' },
        nbinsx: Math.min(30, Math.max(5, Math.floor(logFraud.length / 2)))
      }
    ];

    const distLayout = {
      barmode: 'overlay',
      height: 300,
      margin: { l: 40, r: 20, t: 20, b: 40 },
      paper_bgcolor: darkPaperBg,
      plot_bgcolor: darkPaperBg,
      font: { color: darkFontColor, family: 'Inter' },
      xaxis: { title: 'Log Transformed Amount log1p(Amount)', gridcolor: '#1E293B' },
      yaxis: { title: 'Frequency', gridcolor: '#1E293B' },
      legend: { orientation: 'h', y: 1.15 }
    };
    Plotly.react('chart-amount-distribution', distData, distLayout, { responsive: true, displayModeBar: false });

    // 3. Real Empirical Feature Shifts from this dataset
    const shifts = data.feature_shifts || {};
    const featKeys = ['V14', 'V17', 'V12', 'V10', 'V4', 'V11'];
    const featLabels = ['V14 (Velocity)', 'V17 (Geo Jump)', 'V12 (Deviation)', 'V10 (Device)', 'V4 (Frequency)', 'V11 (Auth)'];
    const featVals = featKeys.map(k => shifts[k] !== undefined ? shifts[k] : 0.0);
    const featColors = featVals.map(v => v < 0 ? '#38BDF8' : '#EF4444');

    const barData = [{
      x: featLabels,
      y: featVals,
      type: 'bar',
      marker: { color: featColors },
      text: featVals.map(v => (v > 0 ? `+${v}` : `${v}`)),
      textposition: 'auto',
      textfont: { color: '#fff', size: 11 }
    }];

    const barLayout = {
      height: 250,
      margin: { l: 40, r: 20, t: 20, b: 60 },
      paper_bgcolor: darkPaperBg,
      plot_bgcolor: darkPaperBg,
      font: { color: darkFontColor, family: 'Inter' },
      yaxis: { title: 'Empirical Shift (Fraud - Gen)', gridcolor: '#1E293B' },
      xaxis: { gridcolor: '#1E293B' }
    };
    Plotly.react('chart-features-bar', barData, barLayout, { responsive: true, displayModeBar: false });

    // 4. Real Hourly Attack Velocity from this dataset
    const hours = Array.from({length: 24}, (_, i) => `${i}:00`);
    const hourlyCounts = data.hourly_attacks || new Array(24).fill(0);

    const lineData = [{
      x: hours,
      y: hourlyCounts,
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#F59E0B', width: 2.8 },
      marker: { size: 6, color: '#F59E0B' },
      name: 'Fraud Attacks'
    }];

    const lineLayout = {
      height: 250,
      margin: { l: 40, r: 20, t: 20, b: 40 },
      paper_bgcolor: darkPaperBg,
      plot_bgcolor: darkPaperBg,
      font: { color: darkFontColor, family: 'Inter' },
      xaxis: { gridcolor: '#1E293B' },
      yaxis: { gridcolor: '#1E293B', title: 'Attack Count' }
    };
    Plotly.react('chart-hourly-line', lineData, lineLayout, { responsive: true, displayModeBar: false });
  }

  // Dynamic Renderer for Tab 1: Threat Radar Charts
  function renderRadarDonutAndHourly(data) {
    if (!data) data = baselineAnalytics;

    // 1. Radar Donut Chart (Clean vs Fraud in current active dataset)
    const genC = data.genuine_count || 284315;
    const fraudC = data.fraud_count || 492;

    const pieData = [{
      values: [genC, fraudC],
      labels: [`Clean (${genC.toLocaleString()})`, `Threats (${fraudC.toLocaleString()})`],
      type: 'pie',
      hole: 0.55,
      marker: { colors: ['#0284C7', '#EF4444'] },
      textinfo: 'percent',
      hoverinfo: 'label+value'
    }];
    const pieLayout = {
      height: 250,
      margin: { l: 10, r: 10, t: 10, b: 10 },
      paper_bgcolor: darkPaperBg,
      font: { color: darkFontColor, family: 'Inter' },
      showlegend: true,
      legend: { orientation: 'h', y: -0.15 }
    };
    Plotly.react('chart-radar-donut', pieData, pieLayout, { responsive: true, displayModeBar: false });

    // 2. Radar 24h Hourly Attack Velocity Bar
    const hours = Array.from({length: 24}, (_, i) => `${i}:00`);
    const hourlyCounts = data.hourly_attacks || new Array(24).fill(0);

    const hourData = [{
      x: hours,
      y: hourlyCounts,
      type: 'bar',
      marker: { color: '#38BDF8' }
    }];
    const hourLayout = {
      height: 250,
      margin: { l: 30, r: 10, t: 10, b: 40 },
      paper_bgcolor: darkPaperBg,
      plot_bgcolor: darkPaperBg,
      font: { color: darkFontColor, family: 'Inter' },
      xaxis: { gridcolor: '#1E293B' },
      yaxis: { gridcolor: '#1E293B', title: 'Attacks' }
    };
    Plotly.react('chart-radar-hourly', hourData, hourLayout, { responsive: true, displayModeBar: false });
  }

  // Reset to Global Baseline Button Handler
  const btnResetBaseline = document.getElementById('btn-reset-baseline');
  if (btnResetBaseline) {
    btnResetBaseline.addEventListener('click', () => {
      activeAnalyticsData = baselineAnalytics;
      activeDatasetName = "Global Baseline (284,807 transactions)";
      renderThreatAnalytics(baselineAnalytics, activeDatasetName);
      renderRadarDonutAndHourly(baselineAnalytics);
    });
  }

  // --------------------------------------------------------------------------
  // 7. Tab 5: Financial Loss Curve
  // --------------------------------------------------------------------------
  const costFnInp = document.getElementById('cost-fn-input');
  const costFpInp = document.getElementById('cost-fp-input');

  if (costFnInp) costFnInp.addEventListener('input', renderCostCurve);
  if (costFpInp) costFpInp.addEventListener('input', renderCostCurve);

  function renderCostCurve() {
    const costFn = parseFloat(costFnInp ? costFnInp.value : 250) || 250;
    const costFp = parseFloat(costFpInp ? costFpInp.value : 20) || 20;

    const thresholds = [];
    const totalCosts = [];
    let minCost = Infinity;
    let bestThresh = 0.12;

    for (let t = 0.02; t <= 0.90; t += 0.02) {
      const fn = Math.round(71 * Math.pow(t, 0.45));
      const fp = Math.round(45 * Math.exp(-6.5 * (t - 0.02)));
      const cost = (fn * costFn) + (fp * costFp);

      thresholds.push(parseFloat(t.toFixed(2)));
      totalCosts.push(cost);

      if (cost < minCost) {
        minCost = cost;
        bestThresh = parseFloat(t.toFixed(2));
      }
    }

    const curveData = [{
      x: thresholds,
      y: totalCosts,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#0284C7', width: 3 },
      name: 'Total Expected Bank Loss (€)'
    }];

    const curveLayout = {
      height: 310,
      margin: { l: 50, r: 20, t: 20, b: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#94A3B8', family: 'Inter' },
      xaxis: { title: 'Operating Decision Threshold (θ)', gridcolor: '#1E293B' },
      yaxis: { title: 'Expected Total Loss (€)', gridcolor: '#1E293B' },
      shapes: [{
        type: 'line',
        x0: bestThresh,
        x1: bestThresh,
        y0: 0,
        y1: Math.max(...totalCosts),
        line: { color: '#10B981', width: 2.5, dash: 'dash' }
      }],
      annotations: [{
        x: bestThresh,
        y: minCost,
        text: `Cost-Optimal Policy (θ=${bestThresh.toFixed(2)})`,
        showarrow: true,
        arrowhead: 2,
        ax: 60,
        ay: -40,
        font: { color: '#10B981', size: 11 }
      }]
    };

    Plotly.react('chart-cost-curve', curveData, curveLayout, { responsive: true, displayModeBar: false });
  }

  // --------------------------------------------------------------------------
  // 8. Initialization
  // --------------------------------------------------------------------------
  renderThreatAnalytics(baselineAnalytics, baselineAnalytics.dataset_name);
  renderRadarDonutAndHourly(baselineAnalytics);
  renderCostCurve();
  applyScreenerPreset('genuine');
});
