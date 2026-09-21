/* ═══════════════════════════════════════════════
   KAIKETSU — app.js  (Frontend Logic)
   All 13 pipeline steps + Canvas visualization
   ═══════════════════════════════════════════════ */

'use strict';

// ── PIPELINE STAGE DEFINITIONS ──────────────────
const PIPELINE_STAGES = [
  { num:'01', icon:'📡', name:'Preprocessing',        tech:'Lee Filter + Calibration + Geocoding',      key:'preprocess' },
  { num:'02', icon:'🌊', name:'Sea-Ice Segmentation', tech:'U-Net (water vs ice mask)',                  key:'segmentation' },
  { num:'03', icon:'🧊', name:'Iceberg Detection',    tech:'YOLO / ResUNet (bbox + confidence)',         key:'detection' },
  { num:'04', icon:'🚢', name:'Ship Discriminator',   tech:'CNN Classifier (ship vs iceberg)',           key:'discriminator' },
  { num:'05', icon:'🚫', name:'Ice Floe Filter',      tech:'ML Classifier + Rules (deadlock avoidance)', key:'floe_filter' },
  { num:'06', icon:'🎯', name:'Iceberg Tracker',      tech:'Kalman Filter + Deep SORT (ID tracking)',    key:'tracker' },
  { num:'07', icon:'💥', name:'Breakup Detector',     tech:'Change Detection + Graph (parent→children)', key:'breakup' },
  { num:'08', icon:'📍', name:'Trajectory Model',     tech:'Physics + LSTM (24h/48h/72h path)',          key:'trajectory' },
  { num:'09', icon:'🌡️', name:'Sea-Ice Forecast',     tech:'ConvLSTM / U-Net Forecaster',               key:'forecast' },
  { num:'10', icon:'🔥', name:'Hazard Fusion',        tech:'GBT Risk Scoring (risk map)',                key:'hazard' },
  { num:'11', icon:'⛽', name:'Vessel Performance',   tech:'NN / GBT (speed & fuel prediction)',         key:'vessel' },
  { num:'12', icon:'🗺️', name:'Route Optimizer',      tech:'A* + NSGA-II Pareto (3 route options)',      key:'router' },
  { num:'13', icon:'📊', name:'Dashboard & ECDIS',    tech:'REST API + JSON (onboard integration)',      key:'dashboard' },
];

const SAMPLE_DATASETS = [
  { name:'Antarctic SAR 1',  desc:'Sentinel-1 · Weddell Sea',  size:'2.4MB' },
  { name:'Antarctic SAR 2',  desc:'Sentinel-1 · Ross Ice Shelf',size:'3.1MB' },
  { name:'Iceberg Cluster',  desc:'CryoSat-2 · Multi-berg',    size:'1.8MB' },
  { name:'AMSR Sea-Ice',     desc:'Microwave · 12.5km res',    size:'890KB' },
];

const KAGGLE_DATASETS = [
  { name:'Sentinel-1 SAR Iceberg (Statoil)',  source:'Kaggle',    size:'~2.5GB' },
  { name:'NSIDC Sea-Ice Concentration',       source:'NSIDC',     size:'~1.8GB' },
  { name:'Antarctic Ice Sheet Altimetry',     source:'CryoSat-2', size:'~900MB' },
  { name:'GFS Weather + Ocean Currents',      source:'Copernicus',size:'~400MB' },
  { name:'AIS Ship Tracking (Polar)',         source:'MarineTraffic',size:'~120MB' },
  { name:'Bathymetric Depth (GEBCO)',         source:'GEBCO',     size:'~600MB' },
];

const AI_MODELS = [
  { name:'Preprocessing CNN Filter', tech:'PyTorch + OpenCV Lee Filter', key:'preprocess' },
  { name:'U-Net Sea-Ice Segmentation',tech:'PyTorch U-Net (encoder-decoder)', key:'segmentation' },
  { name:'YOLO Iceberg Detector',    tech:'Ultralytics YOLOv8 / ResUNet',  key:'detection' },
  { name:'Ship Discriminator CNN',   tech:'PyTorch ResNet-18 Classifier',  key:'discriminator' },
  { name:'Ice Floe ML Filter',       tech:'Scikit-Learn RF + Rules',       key:'floe_filter' },
  { name:'Kalman + DeepSORT Tracker',tech:'FilterPy + DeepSORT',           key:'tracker' },
  { name:'Breakup Graph Detector',   tech:'NetworkX + Change Detection',   key:'breakup' },
  { name:'Physics + LSTM Trajectory',tech:'PyTorch LSTM + ODE Solver',     key:'trajectory' },
  { name:'ConvLSTM Forecast',        tech:'PyTorch ConvLSTM Forecaster',   key:'forecast' },
  { name:'Hazard GBT Fusion',        tech:'XGBoost GBT Risk Scoring',      key:'hazard' },
  { name:'Vessel Performance NN',    tech:'PyTorch NN + XGBoost GBT',      key:'vessel' },
  { name:'A* + NSGA-II Optimizer',   tech:'Heapq A* + DEAP NSGA-II',       key:'router' },
];

// ── STATE ──────────────────────────────────────
const state = {
  selectedSample: null,
  uploadedImage: null,
  pipelineRunning: false,
  pipelineStep: -1,
  modelStatuses: {},
  layers: { ice:true, berg:true, hazard:true, routes:true },
  activeRoute: 'safe',
  icebergs: [],
  downloadPct: 0,
};
AI_MODELS.forEach(m => state.modelStatuses[m.key] = 'idle');

// Synthetic simulated icebergs
const SIM_ICEBERGS = [
  { id:'A-01', lat:-68.42, lon:-52.10, size:1240, conf:0.94, threat:'HIGH',   tag:'REAL ICEBERG',   trajectory:[[320,180],[340,210],[360,250]] },
  { id:'A-02', lat:-67.88, lon:-53.21, size: 340, conf:0.87, threat:'MED',    tag:'REAL ICEBERG',   trajectory:[[220,300],[200,330],[180,360]] },
  { id:'A-03', lat:-68.05, lon:-51.90, size: 820, conf:0.91, threat:'HIGH',   tag:'REAL ICEBERG',   trajectory:[[430,140],[440,170],[455,200]] },
  { id:'S-01', lat:-67.75, lon:-52.80, size: 120, conf:0.12, threat:'IGNORE', tag:'SHIP → IGNORED', trajectory:null },
  { id:'F-01', lat:-68.20, lon:-52.50, size: 450, conf:0.35, threat:'IGNORE', tag:'ICE FLOE → FILTERED', trajectory:null },
];

// ── INIT ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderPipelineStages();
  renderSampleGrid();
  renderDatasetList();
  renderModelList();
  drawHeroCanvas();
  drawNavCanvas();
  renderRouteStats('safe');
  renderForecastFrames();
  renderHazardGauge();
  setupDropZone();
  setupFileInput();
});

// ── UTILITIES ───────────────────────────────────
function scrollTo(sel) {
  const el = document.querySelector(sel);
  if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function rand(min, max) { return Math.random() * (max - min) + min; }

function setStatus(text, type='ready') {
  const dot  = document.getElementById('statusDot');
  const span = document.getElementById('statusText');
  span.textContent = text;
  const colors = { ready:'#06d6a0', running:'#00b4d8', warning:'#ffbe0b', error:'#ff006e' };
  dot.style.background  = colors[type] || colors.ready;
  dot.style.boxShadow   = `0 0 8px ${colors[type] || colors.ready}`;
}

// ── PIPELINE STAGE RENDERING ─────────────────────
function renderPipelineStages() {
  const container = document.getElementById('pipelineStages');
  container.innerHTML = PIPELINE_STAGES.map(s => `
    <div class="stage-card" id="stage_${s.key}">
      <div class="stage-num">MODULE ${s.num}</div>
      <span class="stage-icon">${s.icon}</span>
      <div class="stage-name">${s.name}</div>
      <div class="stage-tech">${s.tech}</div>
    </div>
  `).join('');
}

// ── SAMPLE GRID ──────────────────────────────────
function renderSampleGrid() {
  const grid = document.getElementById('sampleGrid');
  grid.innerHTML = SAMPLE_DATASETS.map((s, i) => `
    <button class="sample-btn" onclick="selectSample(${i})" id="sample_${i}">
      <span class="sample-name">${s.name}</span>
      <span style="font-size:0.7rem;color:#5a7898">${s.desc}</span>
      <span class="sample-size">${s.size}</span>
    </button>
  `).join('');
}

function selectSample(i) {
  state.selectedSample = i;
  document.querySelectorAll('.sample-btn').forEach((b, j) => {
    b.style.borderColor = j === i ? '#00f5d4' : '';
    b.style.color       = j === i ? '#00f5d4' : '';
  });
  const sample = SAMPLE_DATASETS[i];
  setStatus(`Sample loaded: ${sample.name}`, 'ready');
  document.getElementById('canvasOverlay').classList.add('hidden');
  drawSARImageSim(sample);
  buildStepProgress();
}

// ── DEADLOCK MODAL HANDLERS ───────────────────────
function showDeadlockModal(reasonText) {
  const modal = document.getElementById('deadlockModal');
  const reasonBox = document.getElementById('deadlockReason');
  if (reasonBox && reasonText) {
    reasonBox.innerHTML = `⚠️ <strong>Model Error:</strong> ${reasonText}`;
  }
  if (modal) modal.classList.remove('hidden');
  setStatus('Deadlock Error: AI Model Not Trained for this Image', 'error');
}

function dismissDeadlockModal() {
  const modal = document.getElementById('deadlockModal');
  if (modal) modal.classList.add('hidden');
  selectSample(0);
}

// ── FILE INPUT SETUP ─────────────────────────────
function setupFileInput() {
  document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check filename for obvious non-SAR / non-iceberg images
    const fname = file.name.toLowerCase();
    const invalidKeywords = ['cat', 'dog', 'car', 'flower', 'person', 'ship_only', 'avatar', 'logo', 'banner', 'me', 'selfie', 'document'];
    const isInvalidName = invalidKeywords.some(kw => fname.includes(kw));

    state.uploadedImage = file;
    state.selectedSample = null;

    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.getElementById('analysisCanvas');
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Perform pixel variance check for non-SAR colored photos
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        let colorDiffSum = 0;
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 16) {
          const r = data[i], g = data[i+1], b = data[i+2];
          colorDiffSum += Math.abs(r - g) + Math.abs(g - b) + Math.abs(r - b);
        }
        const avgColorVariance = colorDiffSum / (data.length / 16);

        // SAR images are monochromatic/grayscale. High color variance indicates standard RGB photo
        if (isInvalidName || avgColorVariance > 35) {
          showDeadlockModal(`Uploaded file "${file.name}" is a non-polar / non-SAR image (Avg color variance: ${avgColorVariance.toFixed(1)}). The Kaiketsu AI decision model is exclusively trained on Antarctic SAR satellite data.`);
          return;
        }

        document.getElementById('canvasOverlay').classList.add('hidden');
        buildStepProgress();
        setStatus(`Loaded SAR Image: ${file.name}`, 'ready');
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  });
}

// ── DROP ZONE ────────────────────────────────────
function setupDropZone() {
  const dz = document.getElementById('dropZone');
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) document.getElementById('fileInput').files = e.dataTransfer.files;
    document.getElementById('fileInput').dispatchEvent(new Event('change'));
  });
}

// ── SAR IMAGE SIMULATION DRAW ────────────────────
function drawSARImageSim(sample) {
  const canvas = document.getElementById('analysisCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Dark ocean background
  const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W*0.8);
  grad.addColorStop(0, '#0d1f3a');
  grad.addColorStop(1, '#050d1f');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Simulated SAR speckle noise
  for (let i = 0; i < 4000; i++) {
    const x = Math.random()*W, y = Math.random()*H;
    const v = Math.random() * 0.12;
    ctx.fillStyle = `rgba(255,255,255,${v})`;
    ctx.fillRect(x, y, 1.5, 1.5);
  }

  // Sea ice patches
  ctx.fillStyle = 'rgba(0, 180, 216, 0.35)';
  for (let i = 0; i < 6; i++) {
    ctx.beginPath();
    ctx.ellipse(rand(60,W-60), rand(40,H-60), rand(40,120), rand(25,70), rand(0,Math.PI), 0, Math.PI*2);
    ctx.fill();
  }

  // Icebergs
  SIM_ICEBERGS.filter(b => b.threat !== 'IGNORE').forEach(b => {
    const x = b.trajectory[0][0] * W/720, y = b.trajectory[0][1] * H/480;
    ctx.beginPath();
    ctx.arc(x, y, 10 + b.size/200, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(0,245,212,0.7)';
    ctx.fill();
    ctx.strokeStyle = '#00f5d4';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px Outfit, sans-serif';
    ctx.fillText(b.id, x - 10, y - 14);
  });
}

// ── STEP PROGRESS ────────────────────────────────
function buildStepProgress() {
  const container = document.getElementById('stepProgress');
  container.innerHTML = PIPELINE_STAGES.map(s => `
    <div class="step-item" id="step_${s.key}">
      <span class="step-icon">${s.icon}</span>
      <span class="step-label">${s.name}</span>
      <span class="step-status" id="stepStatus_${s.key}">Waiting</span>
    </div>
  `).join('');
}

// ── RUN PIPELINE ─────────────────────────────────
async function runPipeline() {
  if (state.pipelineRunning) return;
  if (!state.selectedSample && !state.uploadedImage) {
    selectSample(0);
    await sleep(300);
  }
  state.pipelineRunning = true;
  setStatus('Pipeline running...', 'running');
  document.getElementById('runBtn').disabled = true;
  document.getElementById('runBtn').textContent = '⏳ Running...';
  buildStepProgress();

  const results = {};

  // If user uploaded a real image, send it to backend analyze endpoint
  if (state.uploadedImage) {
    try {
      const resp = await uploadToAPI(state.uploadedImage);
      if (resp && resp.preprocessing) {
        results.preprocess = resp.preprocessing;
        // If processed image returned, draw it into analysisCanvas
        if (resp.processed_image_base64) {
          const imgEl = new Image();
          imgEl.onload = () => {
            const canvas = document.getElementById('analysisCanvas');
            const ctx = canvas.getContext('2d');
            ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
          };
          imgEl.src = 'data:image/png;base64,' + resp.processed_image_base64;
        }
      }
    } catch (err) {
      console.error('Upload failed', err);
      setStatus('Upload failed', 'error');
    }
  }

  for (let i = 0; i < PIPELINE_STAGES.length; i++) {
    const stage = PIPELINE_STAGES[i];

    // Activate current stage card
    document.querySelectorAll('.stage-card').forEach(c => c.classList.remove('active'));
    const stageCard = document.getElementById(`stage_${stage.key}`);
    if (stageCard) stageCard.classList.add('active');

    const stepEl = document.getElementById(`step_${stage.key}`);
    if (stepEl) {
      stepEl.classList.add('running');
      document.getElementById(`stepStatus_${stage.key}`).textContent = 'Running...';
    }

    await sleep(rand(400, 800));

    // Store simulated result
    results[stage.key] = simulateStageResult(stage.key, results);

    // Mark done
    if (stepEl) {
      stepEl.classList.remove('running');
      stepEl.classList.add('done');
      document.getElementById(`stepStatus_${stage.key}`).textContent = 'Done ✓';
    }
    if (stageCard) stageCard.classList.remove('active');
    if (stageCard) stageCard.classList.add('done');
  }

  // Render 11-Step Streamlit visual stream
  renderStreamlitStream();
  document.getElementById('streamlitStream').style.display = 'block';

  // Show summary results
  renderResults(results);
  setStatus('Pipeline complete!', 'ready');
  document.getElementById('runBtn').disabled = false;
  document.getElementById('runBtn').textContent = '▶ Run Full Pipeline';
  state.pipelineRunning = false;

  // Auto-draw nav canvas with results
  drawNavCanvas(results);
  document.getElementById('resultsGrid').style.display = 'grid';
  scrollTo('#streamlitStream');
}

// Upload helper
async function uploadToAPI(file) {
  const fd = new FormData();
  fd.append('file', file, file.name);
  const res = await fetch('http://localhost:8000/api/analyze-image', { method: 'POST', body: fd });
  if (!res.ok) throw new Error('Upload failed: ' + res.statusText);
  return await res.json();
}

function simulateStageResult(key, prev) {
  switch(key) {
    case 'preprocess':    return { status:'ok', noise_reduction_db: (rand(3,6)).toFixed(1) };
    case 'segmentation':  return { water_pct: (rand(40,60)).toFixed(1), ice_pct:(rand(25,40)).toFixed(1), lead_pct:(rand(5,15)).toFixed(1) };
    case 'detection':     return { icebergs: SIM_ICEBERGS.filter(b=>b.threat!=='IGNORE'), raw_detections: SIM_ICEBERGS.length };
    case 'discriminator': return { ships_removed:1, icebergs_kept: SIM_ICEBERGS.filter(b=>b.threat!=='IGNORE').length };
    case 'floe_filter':   return { floes_filtered:1, icebergs_confirmed: SIM_ICEBERGS.filter(b=>b.threat!=='IGNORE').length };
    case 'tracker':       return { tracks: SIM_ICEBERGS.filter(b=>b.threat!=='IGNORE').map(b=>({id:b.id, frames:rand(5,20).toFixed(0)})) };
    case 'breakup':       return { events:1, parent:'A-03', children:['A-03a','A-03b'] };
    case 'trajectory':    return { predictions: SIM_ICEBERGS.filter(b=>b.trajectory).map(b=>({id:b.id, path:b.trajectory})) };
    case 'forecast':      return { t0:'28%', t24:'34%', t48:'41%', t72:'48%' };
    case 'hazard':        return { zones:[{label:'High Risk',score:0.82},{label:'Medium Risk',score:0.51},{label:'Low Risk',score:0.19}] };
    case 'vessel':        return { speed_kts:12.3, fuel_mt_day:45.2, resistance_kn:18.4 };
    case 'router':        return {
      safe:     { waypoints:12, eta:'38h', fuel:'184mt', risk_score:0.12 },
      balanced: { waypoints:9,  eta:'31h', fuel:'160mt', risk_score:0.35 },
      fast:     { waypoints:7,  eta:'27h', fuel:'148mt', risk_score:0.62 },
    };
    case 'dashboard':     return { api_endpoints:5, ecdis_ready:true };
    default:              return {};
  }
}

// ── RENDER RESULTS ───────────────────────────────
function renderResults(results) {
  // Icebergs
  const icebergList = document.getElementById('icebergList');
  const bergs = SIM_ICEBERGS;
  icebergList.innerHTML = bergs.map(b => {
    const confLevel = b.conf > 0.8 ? 'high' : b.conf > 0.5 ? 'med' : 'low';
    return `
      <div class="berg-item">
        <div class="berg-id">${b.icon || '🧊'} Iceberg ID: ${b.id}
          <span class="berg-conf ${confLevel}">${(b.conf*100).toFixed(0)}% conf</span>
        </div>
        <div class="berg-meta">
          <span>📐 ${b.size} m²</span>
          <span>⚠️ ${b.threat}</span>
          <span>🏷️ ${b.tag}</span>
        </div>
      </div>
    `;
  }).join('');

  // Segmentation canvas
  drawSegmentationCanvas(results.segmentation || {});

  // Avoidance log
  const log = document.getElementById('avoidanceLog');
  log.innerHTML = `
    <div class="avoid-item detected">✅ ICEBERG A-01 (1240m²) — REAL ICEBERG → Tracked (conf: 94%)</div>
    <div class="avoid-item detected">✅ ICEBERG A-02 (340m²)  — REAL ICEBERG → Tracked (conf: 87%)</div>
    <div class="avoid-item detected">✅ ICEBERG A-03 (820m²)  — REAL ICEBERG → Tracked (conf: 91%)</div>
    <div class="avoid-item ignored">🚢 OBJECT S-01 (120m²)  — SHIP DETECTED → Discriminator: IGNORED (conf: 12%)</div>
    <div class="avoid-item ignored">🌊 OBJECT F-01 (450m²)  — ICE FLOE FILTER → IGNORED (not free iceberg)</div>
    <div class="avoid-item warning">⚠️ Breakup Event: A-03 → fragments A-03a, A-03b detected</div>
  `;
}

// ── SEGMENTATION CANVAS ──────────────────────────
function drawSegmentationCanvas(seg) {
  const canvas = document.getElementById('segCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background (water)
  ctx.fillStyle = '#050d1f';
  ctx.fillRect(0, 0, W, H);

  // Sea ice patches
  ctx.globalAlpha = 0.6;
  ctx.fillStyle = '#00b4d8';
  for (let i = 0; i < 4; i++) {
    ctx.beginPath();
    ctx.ellipse(rand(40,W-40), rand(30,H-50), rand(30,80), rand(20,50), rand(0,Math.PI), 0, Math.PI*2);
    ctx.fill();
  }

  // Icebergs
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#00f5d4';
  SIM_ICEBERGS.filter(b=>b.threat!=='IGNORE').forEach(b => {
    const x = b.trajectory[0][0] * W/720;
    const y = b.trajectory[0][1] * H/480;
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI*2);
    ctx.fill();
  });

  // Stats overlay
  ctx.globalAlpha = 0.85;
  ctx.fillStyle = '#020810';
  ctx.fillRect(0, H-50, W, 50);
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#9bb4d4';
  ctx.font = '11px Inter, sans-serif';
  ctx.fillText(`Water: ${seg.water_pct||'52.3'}%  Ice: ${seg.ice_pct||'31.8'}%  Leads: ${seg.lead_pct||'8.4'}%`, 10, H-24);
}

// ── NAV CANVAS (MAP) ─────────────────────────────
function drawNavCanvas(results) {
  const canvas = document.getElementById('navCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Ocean background
  const g = ctx.createLinearGradient(0,0,W,H);
  g.addColorStop(0,'#050d1f'); g.addColorStop(1,'#091628');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(0,100,160,0.15)';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x+=60) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y+=60) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  if (state.layers.ice) {
    // Sea ice concentration blobs
    ctx.globalAlpha = 0.4;
    const iceGrad = ctx.createRadialGradient(300,120,0,300,120,200);
    iceGrad.addColorStop(0,'#00b4d8'); iceGrad.addColorStop(1,'transparent');
    ctx.fillStyle = iceGrad;
    ctx.fillRect(0,0,W,H);

    ctx.globalAlpha = 0.25;
    for (let i = 0; i < 6; i++) {
      ctx.beginPath();
      ctx.ellipse(rand(80,580),rand(30,200),rand(60,140),rand(30,80),rand(0,Math.PI),0,Math.PI*2);
      ctx.fillStyle='rgba(0,180,216,0.5)';
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  if (state.layers.hazard) {
    // Hazard zones (red blobs)
    [[330,200,60],[440,160,80],[260,250,50]].forEach(([x,y,r]) => {
      const hg = ctx.createRadialGradient(x,y,0,x,y,r);
      hg.addColorStop(0,'rgba(255,0,110,0.45)');
      hg.addColorStop(1,'rgba(255,0,110,0)');
      ctx.fillStyle = hg;
      ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
    });
  }

  if (state.layers.berg) {
    // Icebergs + trajectories
    SIM_ICEBERGS.filter(b=>b.threat!=='IGNORE').forEach(b => {
      const pts = b.trajectory;
      // Trajectory arrow
      ctx.strokeStyle = 'rgba(0,245,212,0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6,4]);
      ctx.beginPath();
      pts.forEach((p,i) => i===0 ? ctx.moveTo(p[0],p[1]) : ctx.lineTo(p[0],p[1]));
      ctx.stroke();
      ctx.setLineDash([]);

      // Confidence ring
      const [cx,cy] = pts[0];
      ctx.beginPath(); ctx.arc(cx,cy,18+b.size/200,0,Math.PI*2);
      ctx.strokeStyle='rgba(0,245,212,0.2)'; ctx.lineWidth=8; ctx.stroke();

      // Iceberg dot
      ctx.beginPath(); ctx.arc(cx,cy,8,0,Math.PI*2);
      ctx.fillStyle='#00f5d4'; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();

      // Label
      ctx.fillStyle='#fff';
      ctx.font='bold 11px Outfit,sans-serif';
      ctx.fillText(b.id, cx-10, cy-20);
    });
  }

  if (state.layers.routes) {
    drawRoute(ctx, W, H, state.activeRoute);
  }

  // Ship marker
  ctx.fillStyle='#3a86ff'; ctx.font='22px sans-serif';
  ctx.fillText('🚢', 40, H-40);
  ctx.fillStyle='#9bb4d4'; ctx.font='11px Inter,sans-serif';
  ctx.fillText('MV Maitri', 40, H-20);

  // Destination
  ctx.fillStyle='#ffbe0b'; ctx.font='18px sans-serif';
  ctx.fillText('🏔️', W-60, H-44);
  ctx.fillStyle='#9bb4d4'; ctx.font='10px Inter,sans-serif';
  ctx.fillText('Station', W-62, H-22);
}

function drawRoute(ctx, W, H, type) {
  const routes = {
    safe:     { pts:[[60,H-60],[90,380],[150,320],[200,280],[280,240],[360,200],[430,170],[530,130],[650,90],[W-60,H-44]], color:'#3a86ff', width:3 },
    balanced: { pts:[[60,H-60],[100,360],[180,300],[280,250],[380,210],[460,175],[560,130],[650,90],[W-60,H-44]], color:'#ffbe0b', width:2.5 },
    fast:     { pts:[[60,H-60],[120,320],[240,240],[360,190],[480,150],[600,110],[W-60,H-44]], color:'#fb5607', width:2 },
  };
  const r = routes[type];
  ctx.strokeStyle = r.color;
  ctx.lineWidth = r.width;
  ctx.globalAlpha = type === state.activeRoute ? 1 : 0.35;
  ctx.setLineDash([]);
  ctx.beginPath();
  r.pts.forEach((p,i) => i===0 ? ctx.moveTo(p[0],p[1]) : ctx.lineTo(p[0],p[1]));
  ctx.stroke();

  // Waypoints
  r.pts.forEach(([x,y]) => {
    ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2);
    ctx.fillStyle=r.color; ctx.fill();
  });
  ctx.globalAlpha = 1;
}

// ── LAYER TOGGLES ─────────────────────────────────
function toggleLayer(layer) {
  state.layers[layer] = !state.layers[layer];
  const btn = document.getElementById(`layer${layer.charAt(0).toUpperCase()+layer.slice(1)}`);
  btn?.classList.toggle('active', state.layers[layer]);
  drawNavCanvas();
}

function selectRoute(route) {
  state.activeRoute = route;
  ['safe','balanced','fast'].forEach(r => {
    document.getElementById(`route${r.charAt(0).toUpperCase()+r.slice(1)}`)?.classList.remove('active');
  });
  document.getElementById(`route${route.charAt(0).toUpperCase()+route.slice(1)}`)?.classList.add('active');
  renderRouteStats(route);
  drawNavCanvas();
}

// ── ROUTE STATS ──────────────────────────────────
function renderRouteStats(route) {
  const titles = { safe:'🛡️ Safest Route', balanced:'⚖️ Balanced Route', fast:'⚡ Fastest Route' };
  const data = {
    safe:     [['📍 Waypoints','12'],['⏱️ ETA','38 hours'],['⛽ Fuel Est.','184 metric tons'],['🎯 Risk Score','0.12 (Very Low)'],['📐 Distance','1,840 nm'],['💨 Avg Speed','12.3 kts']],
    balanced: [['📍 Waypoints','9'], ['⏱️ ETA','31 hours'],['⛽ Fuel Est.','160 metric tons'],['🎯 Risk Score','0.35 (Moderate)'], ['📐 Distance','1,740 nm'],['💨 Avg Speed','14.1 kts']],
    fast:     [['📍 Waypoints','7'], ['⏱️ ETA','27 hours'],['⛽ Fuel Est.','148 metric tons'],['🎯 Risk Score','0.62 (High)'],   ['📐 Distance','1,620 nm'],['💨 Avg Speed','16.2 kts']],
  };
  document.getElementById('routeStatTitle').textContent = titles[route];
  document.getElementById('routeStatRows').innerHTML = data[route].map(([l,v]) => `
    <div class="stat-row">
      <span class="stat-row-label">${l}</span>
      <span class="stat-row-value">${v}</span>
    </div>
  `).join('');
}

// ── FORECAST FRAMES ──────────────────────────────
function renderForecastFrames() {
  const frames = [
    {label:'Now (T+0h)',  val:'28% concentration'},
    {label:'T+24h',       val:'34% concentration'},
    {label:'T+48h',       val:'41% concentration'},
    {label:'T+72h',       val:'48% concentration'},
  ];
  document.getElementById('forecastFrames').innerHTML = frames.map(f => `
    <div class="forecast-frame">
      <span class="forecast-label">${f.label}</span>
      <span class="forecast-val">${f.val}</span>
    </div>
  `).join('');
}

// ── HAZARD GAUGE ──────────────────────────────────
function renderHazardGauge() {
  const zones = [
    {label:'Iceberg Collision', score:0.82, color:'#ff006e'},
    {label:'Sea-Ice Entrapment',score:0.51, color:'#ffbe0b'},
    {label:'Shallow Water',     score:0.24, color:'#3a86ff'},
    {label:'Adverse Weather',   score:0.38, color:'#fb5607'},
  ];
  document.getElementById('hazardGauge').innerHTML = zones.map(z => `
    <div class="gauge-row">
      <div class="gauge-label">
        <span class="gauge-name">${z.label}</span>
        <span class="gauge-score">${(z.score*100).toFixed(0)}%</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:${z.score*100}%;background:${z.color}"></div>
      </div>
    </div>
  `).join('');
}

// ── DATASET DOWNLOADER ───────────────────────────
function renderDatasetList() {
  document.getElementById('datasetList').innerHTML = KAGGLE_DATASETS.map((d,i) => `
    <div class="dataset-item" id="dataset_${i}">
      <div>
        <div class="ds-name">${d.name}</div>
        <div class="ds-source">${d.source}</div>
      </div>
      <div class="ds-size">${d.size}</div>
    </div>
  `).join('');
}

async function downloadDatasets() {
  const btn = document.getElementById('downloadBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Downloading...';
  document.getElementById('downloadProgress').style.display = 'block';
  setStatus('Downloading datasets...', 'running');

  const dlBar   = document.getElementById('dlBar');
  const dlLabel = document.getElementById('dlLabel');

  for (let i = 0; i < KAGGLE_DATASETS.length; i++) {
    const ds = KAGGLE_DATASETS[i];
    dlLabel.textContent = `Downloading: ${ds.name} (${ds.source})`;
    document.getElementById(`dataset_${i}`).style.borderColor = '#00b4d8';

    const steps = 15;
    for (let s = 0; s <= steps; s++) {
      dlBar.style.width = `${((i * steps + s) / (KAGGLE_DATASETS.length * steps) * 100).toFixed(1)}%`;
      await sleep(80);
    }
    document.getElementById(`dataset_${i}`).style.borderColor = '#06d6a0';
    addLog(`✅ ${ds.name} — downloaded (${ds.size})`, 'success');
  }

  dlBar.style.width = '100%';
  dlLabel.textContent = 'All datasets ready!';
  btn.disabled = false;
  btn.textContent = '✅ All Downloaded';
  setStatus('Datasets ready. You can now train the AI models.', 'ready');
  addLog('📦 All datasets downloaded. Ready for training.', 'data');
}

// ── MODEL TRAINING ────────────────────────────────
function renderModelList() {
  document.getElementById('modelList').innerHTML = AI_MODELS.map(m => `
    <div class="model-item" id="model_${m.key}">
      <div>
        <div class="model-name">${m.name}</div>
        <div class="model-tech">${m.tech}</div>
      </div>
      <span class="model-status idle" id="modelStatus_${m.key}">Idle</span>
    </div>
  `).join('');
}

async function trainAllModels() {
  addLog('🚀 Starting full AI training run...', 'info');
  setStatus('Training AI models...', 'running');

  for (const model of AI_MODELS) {
    const statusEl = document.getElementById(`modelStatus_${model.key}`);
    if (statusEl) { statusEl.className='model-status training'; statusEl.textContent='Training'; }

    addLog(`🔄 Training: ${model.name} (${model.tech})...`, 'info');

    const epochs = Math.floor(rand(3,8));
    for (let e = 1; e <= epochs; e++) {
      await sleep(rand(400, 900));
      const loss = (rand(0.05, 0.8) * (1 - e/epochs)).toFixed(4);
      const acc  = (rand(75, 99) * e/epochs).toFixed(1);
      addLog(`  Epoch ${e}/${epochs} — loss: ${loss} — acc: ${acc}%`, 'data');
    }

    await sleep(300);
    if (statusEl) { statusEl.className='model-status done'; statusEl.textContent='Done ✓'; }
    addLog(`✅ ${model.name} — Training complete!`, 'success');
    state.modelStatuses[model.key] = 'done';
  }

  setStatus('All models trained!', 'ready');
  addLog('🏁 All 12 AI models trained successfully.', 'success');
}

function resetTraining() {
  AI_MODELS.forEach(m => {
    const el = document.getElementById(`modelStatus_${m.key}`);
    if (el) { el.className='model-status idle'; el.textContent='Idle'; }
    state.modelStatuses[m.key] = 'idle';
  });
  document.getElementById('logTerminal').innerHTML = `
    <div class="log-line log-info">$ Kaiketsu AI Training Console v1.0</div>
    <div class="log-line log-info">$ Training reset. Waiting for command...</div>
  `;
  setStatus('Training reset', 'ready');
}

function addLog(msg, type='info') {
  const terminal = document.getElementById('logTerminal');
  const line = document.createElement('div');
  line.className = `log-line log-${type}`;
  const ts = new Date().toLocaleTimeString('en-IN', {hour12:false});
  line.textContent = `[${ts}] ${msg}`;
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
}

// ── HERO CANVAS ───────────────────────────────────
function drawHeroCanvas() {
  const canvas = document.getElementById('heroCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  function draw(t) {
    ctx.clearRect(0, 0, W, H);

    // Dark polar ocean
    const g = ctx.createLinearGradient(0,0,W,H);
    g.addColorStop(0,'#0a1628'); g.addColorStop(1,'#050d1f');
    ctx.fillStyle = g;
    ctx.fillRect(0,0,W,H);

    // Speckle noise
    for (let i = 0; i < 2500; i++) {
      ctx.fillStyle = `rgba(255,255,255,${Math.random()*0.07})`;
      ctx.fillRect(Math.random()*W,Math.random()*H,1.5,1.5);
    }

    // Pulsing sea ice
    ctx.globalAlpha = 0.35 + 0.08*Math.sin(t*0.5);
    for (let i = 0; i < 8; i++) {
      const px = 80 + (i*55), py = 60 + (i*20)*Math.cos(i*1.1);
      ctx.beginPath();
      ctx.ellipse(px, py+20*Math.sin(t*0.3+i), 55+15*Math.sin(i+t*0.2), 28+8*Math.cos(i+t*0.4), i*0.4, 0, Math.PI*2);
      ctx.fillStyle='rgba(0,180,216,0.6)';
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Animated icebergs
    const bergs = [
      {x:180, y:260, r:22, label:'A-01', color:'#00f5d4'},
      {x:350, y:320, r:14, label:'A-02', color:'#48cae4'},
      {x:420, y:190, r:18, label:'A-03', color:'#00f5d4'},
    ];
    bergs.forEach((b,i) => {
      const bx = b.x + 5*Math.sin(t*0.6+i*1.2);
      const by = b.y + 4*Math.cos(t*0.5+i*0.9);
      // Glow
      const gw = ctx.createRadialGradient(bx,by,0,bx,by,b.r*3);
      gw.addColorStop(0,`rgba(0,245,212,0.2)`); gw.addColorStop(1,'transparent');
      ctx.fillStyle=gw; ctx.beginPath(); ctx.arc(bx,by,b.r*3,0,Math.PI*2); ctx.fill();
      // Berg
      ctx.beginPath(); ctx.arc(bx,by,b.r,0,Math.PI*2);
      ctx.fillStyle=b.color; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();
      // Label
      ctx.fillStyle='#fff'; ctx.font='bold 12px Outfit,sans-serif';
      ctx.fillText(b.label, bx-12, by-b.r-8);
// ── RENDER 11-STEP STREAMLIT VISUAL STREAM ─────────────────
function renderStreamlitStream() {
  const bergs = [
    { id: 'Iceberg 1', x: 420, y: 120, w: 100, h: 80, label: 'Iceberg 1' },
    { id: 'Iceberg 2', x: 160, y: 190, w: 110, h: 90, label: 'Iceberg 2' }
  ];

  // Step 1: Satellite Input (Raw SAR Image)
  const c1 = document.getElementById('vCanvas1');
  if (c1) {
    const ctx = c1.getContext('2d');
    const W = c1.width, H = c1.height;
    ctx.fillStyle = '#050d1f'; ctx.fillRect(0, 0, W, H);
    for (let i = 0; i < 3500; i++) {
      ctx.fillStyle = `rgba(255,255,255,${Math.random()*0.15})`;
      ctx.fillRect(Math.random()*W, Math.random()*H, 1.5, 1.5);
    }
    // High backscatter ice regions
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    bergs.forEach(b => {
      ctx.beginPath(); ctx.ellipse(b.x, b.y, b.w/2, b.h/2, 0.2, 0, Math.PI*2); ctx.fill();
    });
  }

  // Step 2: Preprocessing (Lee filter + contrast boost)
  const c2 = document.getElementById('vCanvas2');
  if (c2) {
    const ctx = c2.getContext('2d');
    const W = c2.width, H = c2.height;
    ctx.fillStyle = '#050d1f'; ctx.fillRect(0, 0, W, H);
    // Smooth background
    const g = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W*0.7);
    g.addColorStop(0, '#0a1a32'); g.addColorStop(1, '#020810');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    bergs.forEach(b => {
      ctx.beginPath(); ctx.ellipse(b.x, b.y, b.w/2, b.h/2, 0.2, 0, Math.PI*2); ctx.fill();
    });
  }

  // Step 3: Segmentation (U-Net water vs ice binary mask)
  const c3 = document.getElementById('vCanvas3');
  if (c3) {
    const ctx = c3.getContext('2d');
    const W = c3.width, H = c3.height;
    ctx.fillStyle = '#0a1628'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#00b4d8';
    bergs.forEach(b => {
      ctx.beginPath(); ctx.ellipse(b.x, b.y, b.w/2, b.h/2, 0.2, 0, Math.PI*2); ctx.fill();
    });
  }

  // Step 4: Iceberg Detection (Red Bounding Boxes)
  const c4 = document.getElementById('vCanvas4');
  if (c4) {
    const ctx = c4.getContext('2d');
    const W = c4.width, H = c4.height;
    ctx.fillStyle = '#050d1f'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    bergs.forEach(b => {
      ctx.beginPath(); ctx.ellipse(b.x, b.y, b.w/2, b.h/2, 0.2, 0, Math.PI*2); ctx.fill();
      // Red Bbox
      ctx.strokeStyle = '#ff0044'; ctx.lineWidth = 3;
      ctx.strokeRect(b.x - b.w/2 - 10, b.y - b.h/2 - 10, b.w + 20, b.h + 20);
      // Red Label
      ctx.fillStyle = '#ff0044'; ctx.font = 'bold 12px Outfit, sans-serif';
      ctx.fillText(b.label, b.x - b.w/2 - 10, b.y - b.h/2 - 16);
    });
  }

  // Step 6: Trajectory (Physics vector paths)
  const c6 = document.getElementById('vCanvas6');
  if (c6) {
    const ctx = c6.getContext('2d');
    const W = c6.width, H = c6.height;
    ctx.fillStyle = '#050d1f'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    bergs.forEach((b, idx) => {
      ctx.beginPath(); ctx.ellipse(b.x, b.y, b.w/2, b.h/2, 0.2, 0, Math.PI*2); ctx.fill();
      // Yellow Trajectory line
      ctx.strokeStyle = '#ffbe0b'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(b.x, b.y);
      const tx = idx === 0 ? b.x + 90 : b.x - 70;
      const ty = idx === 0 ? b.y + 70 : b.y - 70;
      ctx.lineTo(tx, ty); ctx.stroke();
      // Arrow head
      ctx.fillStyle = '#ffbe0b'; ctx.beginPath(); ctx.arc(tx, ty, 6, 0, Math.PI*2); ctx.fill();
      // Bbox
      ctx.strokeStyle = '#ff0044'; ctx.lineWidth = 2;
      ctx.strokeRect(b.x - b.w/2 - 10, b.y - b.h/2 - 10, b.w + 20, b.h + 20);
    });
  }

  // Step 8: Hazard Fusion (Concentric red danger rings)
  const c8 = document.getElementById('vCanvas8');
  if (c8) {
    const ctx = c8.getContext('2d');
    const W = c8.width, H = c8.height;
    ctx.fillStyle = '#0a1628'; ctx.fillRect(0, 0, W, H);
    bergs.forEach(b => {
      // Red danger circles
      [40, 70, 100].forEach(r => {
        ctx.strokeStyle = 'rgba(255, 0, 110, 0.6)'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(b.x, b.y, r, 0, Math.PI*2); ctx.stroke();
      });
      ctx.fillStyle = '#fff'; ctx.fillRect(b.x - 6, b.y - 6, 12, 12);
    });
  }

  // Step 9: Route Optimization (Green route bypassing danger zones)
  const c9 = document.getElementById('vCanvas9');
  if (c9) {
    const ctx = c9.getContext('2d');
    const W = c9.width, H = c9.height;
    ctx.fillStyle = '#0a1628'; ctx.fillRect(0, 0, W, H);
    // Danger zones
    bergs.forEach(b => {
      [40, 70, 100].forEach(r => {
        ctx.strokeStyle = 'rgba(255, 0, 110, 0.4)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(b.x, b.y, r, 0, Math.PI*2); ctx.stroke();
      });
      ctx.fillStyle = '#fff'; ctx.fillRect(b.x - 5, b.y - 5, 10, 10);
    });
    // Start & Destination markers
    const startX = 60, startY = H - 40;
    const destX = W - 60, destY = 50;

    // Green Safe Route
    ctx.strokeStyle = '#06d6a0'; ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(180, H - 70);
    ctx.lineTo(300, H - 50);
    ctx.lineTo(440, H - 120);
    ctx.lineTo(destX, destY);
    ctx.stroke();

    // Start Badge
    ctx.fillStyle = '#06d6a0'; ctx.beginPath(); ctx.arc(startX, startY, 8, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 10px Outfit'; ctx.fillText('START', startX - 16, startY + 20);

    // Dest Badge
    ctx.fillStyle = '#00f5d4'; ctx.beginPath(); ctx.arc(destX, destY, 8, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 10px Outfit'; ctx.fillText('DESTINATION', destX - 30, destY - 14);
  }
}
      ctx.setLineDash([5,5]);
      ctx.strokeStyle='rgba(0,245,212,0.4)'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(bx,by);
      ctx.lineTo(bx+40*Math.cos(t*0.1+i),by+40*Math.sin(t*0.15+i));
      ctx.stroke(); ctx.setLineDash([]);
    });

    // Animated optimal route
    const routeProgress = (t*0.15) % 1;
    ctx.strokeStyle='#3a86ff'; ctx.lineWidth=2.5; ctx.setLineDash([]);
    ctx.beginPath();
    const routePts=[[40,460],[100,400],[180,340],[260,290],[360,250],[450,200],[500,150]];
    routePts.forEach((p,i)=>i===0?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1]));
    ctx.stroke();

    // Moving ship
    const shipIdx = routeProgress*(routePts.length-1);
    const si = Math.floor(shipIdx);
    if (si < routePts.length-1) {
      const frac = shipIdx-si;
      const sx = routePts[si][0]+(routePts[si+1][0]-routePts[si][0])*frac;
      const sy = routePts[si][1]+(routePts[si+1][1]-routePts[si][1])*frac;
      ctx.font='20px sans-serif'; ctx.fillText('🚢',sx-10,sy+8);
    }

    // Risk zones
    [[300,300,50],[380,240,40]].forEach(([x,y,r])=>{
      const rg=ctx.createRadialGradient(x,y,0,x,y,r);
      rg.addColorStop(0,'rgba(255,0,110,0.35)'); rg.addColorStop(1,'transparent');
      ctx.fillStyle=rg; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
    });

    // Corner HUD
    ctx.fillStyle='rgba(10,22,40,0.8)';
    ctx.roundRect(10,10,200,80,10); ctx.fill();
    ctx.fillStyle='#00f5d4'; ctx.font='bold 12px Outfit,sans-serif';
    ctx.fillText('🛰️ KAIKETSU — LIVE',18,30);
    ctx.fillStyle='#9bb4d4'; ctx.font='11px Inter,sans-serif';
    ctx.fillText(`Icebergs: 3 tracked`,18,50);
    ctx.fillText(`Route: Safest (A* + NSGA-II)`,18,66);
    ctx.fillText(`Risk: 0.12 | ETA: 38h`,18,82);

    requestAnimationFrame(t2 => draw(t2/1000));
  }
  requestAnimationFrame(t => draw(t/1000));
}
