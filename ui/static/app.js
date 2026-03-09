const state = {
  jobId: null,
  logCursor: 0,
  pollTimer: null,
  running: false,
  modelCheckId: null,
  modelCheckCursor: 0,
  modelCheckTimer: null,
};

const el = {
  btnCheckModel: document.getElementById('btnCheckModel'),
  btnCheckEnv: document.getElementById('btnCheckEnv'),
  modelCheckList: document.getElementById('modelCheckList'),
  modelCheckLog: document.getElementById('modelCheckLog'),
  envCheckList: document.getElementById('envCheckList'),
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  btnPickFile: document.getElementById('btnPickFile'),
  uploadInfo: document.getElementById('uploadInfo'),
  btnStart: document.getElementById('btnStart'),
  langInput: document.getElementById('langInput'),
  btnClearLog: document.getElementById('btnClearLog'),
  progressText: document.getElementById('progressText'),
  progressPercent: document.getElementById('progressPercent'),
  progressBar: document.getElementById('progressBar'),
  stepList: document.getElementById('stepList'),
  logBox: document.getElementById('logBox'),
  downloadArea: document.getElementById('downloadArea'),
  fullTxtDownload: document.getElementById('fullTxtDownload'),
  segmentsTxtDownload: document.getElementById('segmentsTxtDownload'),
  jsonDownload: document.getElementById('jsonDownload'),
  btnRefreshTemp: document.getElementById('btnRefreshTemp'),
  btnClearTemp: document.getElementById('btnClearTemp'),
  tempSummary: document.getElementById('tempSummary'),
  tempList: document.getElementById('tempList'),
};

function appendLog(line, level = 'info') {
  const row = document.createElement('div');
  if (level === 'success') row.className = 'ok';
  if (level === 'error') row.className = 'err';
  row.textContent = `[${line.ts}] ${line.message}`;
  el.logBox.appendChild(row);
  el.logBox.scrollTop = el.logBox.scrollHeight;
}

function resetLogs() {
  el.logBox.innerHTML = '';
  state.logCursor = 0;
  renderProgress(null);
}

function clearCurrentLogsView() {
  el.logBox.innerHTML = '';
}

function clearList(listEl) {
  listEl.innerHTML = '';
}

function renderStatusList(listEl, items = []) {
  clearList(listEl);
  for (const item of items) {
    const li = document.createElement('li');
    li.className = `status-item ${item.ok ? 'ok' : 'bad'}`;

    const head = document.createElement('div');
    head.className = 'status-head';
    head.textContent = `${item.ok ? '✓' : '✗'} ${item.name || item.key || '检查项'}`;

    const detail = document.createElement('div');
    detail.className = 'status-detail';
    const expected = item.expected ? `期望: ${item.expected}` : '';
    const actual = item.actual ? `实际: ${item.actual}` : '';
    const extra = item.detail || '';
    detail.textContent = [expected, actual, extra].filter(Boolean).join(' | ');

    li.appendChild(head);
    li.appendChild(detail);
    listEl.appendChild(li);
  }
}

function appendModelCheckLog(line, level = 'info') {
  const prefix = level === 'error' ? '[ERR]' : level === 'success' ? '[OK]' : '[INFO]';
  el.modelCheckLog.textContent += `[${line.ts}] ${prefix} ${line.message}\n`;
  el.modelCheckLog.scrollTop = el.modelCheckLog.scrollHeight;
}

function renderProgress(progress) {
  if (!progress) {
    el.progressText.textContent = '等待开始';
    el.progressPercent.textContent = '0.00%';
    el.progressBar.style.width = '0%';
    el.stepList.innerHTML = '';
    return;
  }

  const total = progress.total_parts || 0;
  const done = progress.completed_parts || 0;
  const percent = Number(progress.percent || 0);

  el.progressText.textContent = total > 0 ? `转录进度 ${done}/${total}` : '等待切割结果';
  el.progressPercent.textContent = `${percent.toFixed(2)}%`;
  el.progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;

  el.stepList.innerHTML = '';
  for (const step of progress.steps || []) {
    const item = document.createElement('div');
    item.className = `step-item ${step.status || 'waiting'}`;
    item.textContent = step.detail ? `${step.name} (${step.detail})` : step.name;
    el.stepList.appendChild(item);
  }
}

async function api(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.ok === false) {
    throw new Error(data.error || `请求失败: ${resp.status}`);
  }
  return data;
}

async function checkModel() {
  el.modelCheckLog.textContent = '';
  clearList(el.modelCheckList);
  stopModelCheckPolling();
  el.btnCheckModel.disabled = true;
  try {
    const data = await api('/api/check/model/start', { method: 'POST' });
    state.modelCheckId = data.check_id;
    state.modelCheckCursor = 0;
    appendModelCheckLog({ ts: 'SYS', message: `检测任务已创建: ${state.modelCheckId}` }, 'info');
    startModelCheckPolling();
  } catch (err) {
    el.btnCheckModel.disabled = false;
    appendModelCheckLog({ ts: 'ERROR', message: `检测失败: ${err.message}` }, 'error');
  }
}

async function pollModelCheck() {
  if (!state.modelCheckId) return;
  try {
    const data = await api(`/api/check/model/status/${state.modelCheckId}?since=${state.modelCheckCursor}`);
    for (const line of data.logs || []) {
      appendModelCheckLog(line, line.level || 'info');
    }
    state.modelCheckCursor = data.next_cursor || state.modelCheckCursor;

    if (data.result?.items) {
      renderStatusList(el.modelCheckList, data.result.items);
    }

    if (data.status === 'completed' || data.status === 'failed') {
      stopModelCheckPolling();
      el.btnCheckModel.disabled = false;
      if (data.status === 'failed') {
        appendModelCheckLog({ ts: 'ERROR', message: data.error || '模型检测失败' }, 'error');
      }
    }
  } catch (err) {
    stopModelCheckPolling();
    el.btnCheckModel.disabled = false;
    appendModelCheckLog({ ts: 'ERROR', message: `模型检测轮询失败: ${err.message}` }, 'error');
  }
}

function startModelCheckPolling() {
  stopModelCheckPolling();
  state.modelCheckTimer = setInterval(pollModelCheck, 1000);
}

function stopModelCheckPolling() {
  if (state.modelCheckTimer) {
    clearInterval(state.modelCheckTimer);
    state.modelCheckTimer = null;
  }
}

async function checkEnv() {
  clearList(el.envCheckList);
  try {
    const data = await api('/api/check/env');
    renderStatusList(el.envCheckList, data.items || []);
  } catch (err) {
    renderStatusList(el.envCheckList, [{ name: '环境检测请求', ok: false, detail: err.message }]);
  }
}

async function refreshTempUsage() {
  try {
    const data = await api('/api/temp/usage');
    el.tempSummary.textContent = `临时文件总大小: ${data.total_human}`;
    const items = (data.details || []).map((d) => ({
      name: d.path,
      ok: true,
      detail: `${d.human} (${d.bytes} bytes)`,
    }));
    renderStatusList(el.tempList, items);
  } catch (err) {
    el.tempSummary.textContent = `统计失败: ${err.message}`;
    renderStatusList(el.tempList, [{ name: '临时文件统计', ok: false, detail: err.message }]);
  }
}

async function clearTempFiles() {
  if (!confirm('确认清除当前工作区内 UI 历史临时文件吗？')) return;
  try {
    const data = await api('/api/temp/clear', { method: 'POST' });
    el.tempSummary.textContent = `清理完成，已删除 ${data.removed_entries} 项`;
    renderStatusList(el.tempList, []);
    refreshTempUsage();
  } catch (err) {
    el.tempSummary.textContent = `清理失败: ${err.message}`;
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (file.size > 2 * 1024 * 1024 * 1024) {
    alert('文件超过 2GB 限制');
    return;
  }

  const fd = new FormData();
  fd.append('file', file);

  el.uploadInfo.textContent = `正在上传: ${file.name}`;
  el.btnStart.disabled = true;

  try {
    const data = await api('/api/upload', { method: 'POST', body: fd });
    state.jobId = data.job_id;
    resetLogs();
    el.uploadInfo.textContent = `上传成功: ${data.filename} | job_id=${state.jobId}`;
    el.btnStart.disabled = false;
    el.downloadArea.classList.add('hidden');
  } catch (err) {
    el.uploadInfo.textContent = `上传失败: ${err.message}`;
  }
}

async function startJob() {
  if (!state.jobId || state.running) return;

  resetLogs();
  state.running = true;
  el.btnStart.disabled = true;
  el.downloadArea.classList.add('hidden');

  try {
    await api('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_id: state.jobId,
        language: el.langInput.value.trim() || 'zh',
      }),
    });
    startPolling();
  } catch (err) {
    state.running = false;
    el.btnStart.disabled = false;
    appendLog({ ts: 'ERROR', message: err.message }, 'error');
  }
}

async function pollLogs() {
  if (!state.jobId) return;
  try {
    const data = await api(`/api/jobs/${state.jobId}/logs?since=${state.logCursor}`);
    for (const line of data.logs || []) {
      appendLog(line, line.level || 'info');
    }
    state.logCursor = data.next_cursor || state.logCursor;
    renderProgress(data.progress || null);

    if (data.status === 'completed') {
      stopPolling();
      state.running = false;
      el.btnStart.disabled = false;
      const fullTxtUrl = `/api/download/${state.jobId}/full_txt`;
      const segTxtUrl = `/api/download/${state.jobId}/segments_txt`;
      const jsonUrl = `/api/download/${state.jobId}/json`;
      el.fullTxtDownload.href = fullTxtUrl;
      el.segmentsTxtDownload.href = segTxtUrl;
      el.jsonDownload.href = jsonUrl;
      el.downloadArea.classList.remove('hidden');
      refreshTempUsage();
    } else if (data.status === 'failed') {
      stopPolling();
      state.running = false;
      el.btnStart.disabled = false;
      appendLog({ ts: 'ERROR', message: data.error || '任务失败' }, 'error');
    }
  } catch (err) {
    stopPolling();
    state.running = false;
    el.btnStart.disabled = false;
    appendLog({ ts: 'ERROR', message: `日志轮询失败: ${err.message}` }, 'error');
  }
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(pollLogs, 1000);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function bindUploadEvents() {
  el.btnPickFile.addEventListener('click', () => el.fileInput.click());
  el.fileInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    uploadFile(file);
  });

  const prevent = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((name) => {
    el.dropZone.addEventListener(name, prevent, false);
  });

  ['dragenter', 'dragover'].forEach((name) => {
    el.dropZone.addEventListener(name, () => el.dropZone.classList.add('dragover'));
  });

  ['dragleave', 'drop'].forEach((name) => {
    el.dropZone.addEventListener(name, () => el.dropZone.classList.remove('dragover'));
  });

  el.dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer?.files?.[0];
    uploadFile(file);
  });
}

function init() {
  el.btnCheckModel.addEventListener('click', checkModel);
  el.btnCheckEnv.addEventListener('click', checkEnv);
  el.btnStart.addEventListener('click', startJob);
  el.btnClearLog.addEventListener('click', clearCurrentLogsView);
  el.btnRefreshTemp.addEventListener('click', refreshTempUsage);
  el.btnClearTemp.addEventListener('click', clearTempFiles);
  bindUploadEvents();
  renderProgress(null);
  refreshTempUsage();
}

init();
