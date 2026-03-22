// ── 配置 ─────────────────────────────────────────────────────────────────
const WS_URL    = 'ws://localhost:8000/ws/speech';
const API_BASE  = 'http://localhost:8000/api';
const SAMPLE_RATE   = 16000;
const CHUNK_SAMPLES = 4096;  // ~256ms per send
const SPK_COLORS = ['#58a6ff','#3fb950','#d29922','#bc8cff','#ff7b72','#79c0ff'];

// ── 状态 ─────────────────────────────────────────────────────────────────
let ws = null;
let mediaStream = null;
let audioCtx = null;
let processor = null;
let sessionId = null;
let isPaused = false;
let segCount = 0;
let speakerSet = new Set();
let durationTimer = null;
let elapsedSec = 0;

// ── 波形可视化 ────────────────────────────────────────────────────────────
const canvas = document.getElementById('waveCanvas');
const ctx2d = canvas.getContext('2d');
let waveData = new Float32Array(128).fill(0);

function drawWave() {
  canvas.width = canvas.offsetWidth * devicePixelRatio;
  canvas.height = canvas.offsetHeight * devicePixelRatio;
  canvas.style.width = canvas.offsetWidth + 'px';
  const W = canvas.width, H = canvas.height;
  ctx2d.clearRect(0, 0, W, H);
  const barW = W / waveData.length;
  const color = isPaused ? '#d29922' : '#58a6ff';
  waveData.forEach((v, i) => {
    const h = Math.max(2, Math.abs(v) * H * 0.85);
    ctx2d.fillStyle = color;
    ctx2d.globalAlpha = 0.65 + Math.abs(v) * 0.35;
    ctx2d.fillRect(i * barW, (H - h) / 2, Math.max(1, barW - 1), h);
  });
  requestAnimationFrame(drawWave);
}
drawWave();

// ── WebSocket ─────────────────────────────────────────────────────────────
function connectWS(onOpen) {
  ws = new WebSocket(WS_URL);
  ws.onopen = onOpen;
  ws.onmessage = (e) => handleServerMsg(JSON.parse(e.data));
  ws.onerror = () => setStatus('连接错误', '');
  ws.onclose = () => setStatus('已断开', '');
}

function handleServerMsg(msg) {
  switch (msg.type) {
    case 'session_started':
      sessionId = msg.session_id;
      document.getElementById('infoSid').textContent = msg.session_id.slice(0,8) + '…';
      setStatus('录音中', 'recording');
      break;

    case 'transcript':
      addSegment(msg);
      segCount = msg.seq;
      document.getElementById('infoSegs').textContent = segCount;
      speakerSet.add(msg.speaker_label);
      document.getElementById('infoSpks').textContent = speakerSet.size;
      updateSpeakerEditor();
      break;

    case 'session_completed':
      setStatus('已完成', 'done');
      stopDurationTimer();
      showDownloadLinks(msg.audio_url, msg.transcript_url);
      break;

    case 'paused':  setStatus('已暂停', 'paused'); break;
    case 'resumed': setStatus('录音中', 'recording'); break;
    case 'error':   console.error('[WS Error]', msg.message); break;
  }
}

// ── 录音控制 ──────────────────────────────────────────────────────────────
async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
  } catch (e) {
    alert('无法访问麦克风: ' + e.message);
    return;
  }

  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const source = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(CHUNK_SAMPLES, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN || isPaused) return;
    const samples = e.inputBuffer.getChannelData(0);
    // 更新波形
    const step = Math.floor(samples.length / 128);
    for (let i = 0; i < 128; i++) waveData[i] = samples[i * step];
    // float32 → int16 → base64
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++)
      int16[i] = Math.max(-32768, Math.min(32767, samples[i] * 32768));
    const bytes = new Uint8Array(int16.buffer);
    let binary = '';
    // 分块 btoa 避免栈溢出
    for (let i = 0; i < bytes.length; i += 8192)
      binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    ws.send(JSON.stringify({ type: 'audio', data: btoa(binary) }));
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);

  // 连接 WebSocket
  connectWS(() => {
    const sid = crypto.randomUUID();
    ws.send(JSON.stringify({ type: 'start', session_id: sid, title: '录音_' + sid.slice(0,8) }));
  });

  // 更新 UI
  setBtn('btnRecord', true);
  setBtn('btnPause', false);
  setBtn('btnStop', false);
  isPaused = false;
  elapsedSec = 0;
  startDurationTimer();
}

function togglePause() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  isPaused = !isPaused;
  ws.send(JSON.stringify({ type: isPaused ? 'pause' : 'resume' }));
  const btn = document.getElementById('btnPause');
  btn.textContent = isPaused ? '▶ 继续' : '⏸ 暂停';
  if (isPaused) stopDurationTimer();
  else startDurationTimer();
}

function stopRecording() {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: 'stop' }));
  if (processor) { processor.disconnect(); processor = null; }
  if (audioCtx)  { audioCtx.close(); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  stopDurationTimer();
  waveData.fill(0);
  setBtn('btnRecord', false);
  setBtn('btnPause', true);
  setBtn('btnStop', true);
  document.getElementById('btnPause').textContent = '⏸ 暂停';
  isPaused = false;
}

// ── 计时器 ────────────────────────────────────────────────────────────────
function startDurationTimer() {
  stopDurationTimer();
  durationTimer = setInterval(() => {
    elapsedSec++;
    document.getElementById('infoDur').textContent = elapsedSec + 's';
  }, 1000);
}
function stopDurationTimer() {
  if (durationTimer) { clearInterval(durationTimer); durationTimer = null; }
}

// ── 转写片段渲染 ──────────────────────────────────────────────────────────
function spkColor(label) {
  const idx = parseInt((label || '').replace(/\D/g, '') || '0') % SPK_COLORS.length;
  return SPK_COLORS[idx];
}

function addSegment(msg) {
  const box = document.getElementById('transcriptBox');
  // 移除空提示
  const placeholder = box.querySelector('.placeholder');
  if (placeholder) placeholder.remove();

  const color   = spkColor(msg.speaker_label);
  const display = msg.speaker_display || msg.speaker_label;
  const timeStr = fmtTime(msg.start) + ' – ' + fmtTime(msg.end);

  const div = document.createElement('div');
  div.className = 'seg';
  div.dataset.spk = msg.speaker_label;
  div.innerHTML =
    `<div class="seg-meta">` +
      `<span class="seg-spk" style="background:${color}">${escHtml(display)}</span>` +
      `<span class="seg-time">${timeStr}</span>` +
    `</div>` +
    `<div class="seg-text" style="border-left-color:${color}">${escHtml(msg.text)}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function fmtTime(sec) {
  if (sec == null) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return m + ':' + s;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function clearTranscript() {
  const box = document.getElementById('transcriptBox');
  box.innerHTML = '';
  segCount = 0;
  speakerSet.clear();
  document.getElementById('infoSegs').textContent = '0';
  document.getElementById('infoSpks').textContent = '0';
}

// ── 说话人编辑 ────────────────────────────────────────────────────────────
function updateSpeakerEditor() {
  const editor = document.getElementById('spkEditor');
  // 保留已有 input 的值
  const existing = {};
  editor.querySelectorAll('.spk-input').forEach(el => {
    existing[el.dataset.spk] = el.value;
  });
  editor.innerHTML = '';
  [...speakerSet].sort().forEach(label => {
    const color = spkColor(label);
    const val   = existing[label] !== undefined ? existing[label] : '';
    const row = document.createElement('div');
    row.className = 'spk-row';
    row.innerHTML =
      `<span class="spk-label" style="background:${color}">${escHtml(label)}</span>` +
      `<input class="spk-input" data-spk="${label}" value="${escHtml(val)}" placeholder="真实姓名" />`;
    editor.appendChild(row);
  });
}

async function saveSpeakerNames() {
  if (!sessionId) { alert('请先开始录音'); return; }
  const mapping = {};
  document.querySelectorAll('.spk-input').forEach(inp => {
    const v = inp.value.trim();
    if (v) mapping[inp.dataset.spk] = v;
  });
  if (!Object.keys(mapping).length) { alert('请输入至少一个姓名'); return; }
  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/speakers`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mapping }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    // 实时更新已渲染片段的说话人名称
    Object.entries(mapping).forEach(([label, name]) => {
      document.querySelectorAll(`.seg[data-spk="${label}"] .seg-spk`)
        .forEach(el => { el.textContent = name; });
    });
    alert('保存成功 ✓');
  } catch (e) {
    alert('保存失败: ' + e.message);
  }
}

// ── 下载链接 ─────────────────────────────────────────────────────────────
function showDownloadLinks(audioUrl, textUrl) {
  const sec     = document.getElementById('dlSection');
  const pending = document.getElementById('dlPending');
  let shown = false;
  if (audioUrl) { document.getElementById('dlAudio').href = audioUrl; shown = true; }
  if (textUrl)  { document.getElementById('dlText').href  = textUrl;  shown = true; }
  if (shown) { sec.classList.add('visible'); pending.style.display = 'none'; return; }
  // 回退: 通过 REST 拉取
  if (sessionId) {
    fetch(`${API_BASE}/sessions/${sessionId}/download`)
      .then(r => r.json())
      .then(data => {
        if (data.audio_url)      { document.getElementById('dlAudio').href = data.audio_url; shown = true; }
        if (data.transcript_url) { document.getElementById('dlText').href  = data.transcript_url; shown = true; }
        if (shown) { sec.classList.add('visible'); pending.style.display = 'none'; }
      }).catch(() => {});
  }
}

// ── 工具 ─────────────────────────────────────────────────────────────────
function setStatus(text, cls) {
  const b = document.getElementById('statusBadge');
  b.textContent = text;
  b.className = cls || '';
}

function setBtn(id, disabled) {
  document.getElementById(id).disabled = disabled;
}
