(function () {
  'use strict';

  let currentTaskId = null;
  let pollInterval = null;
  const POLL_MS = 1500;

  function api(path, options) {
    return fetch(path, options).then(function (r) {
      if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || r.statusText); });
      return r.json();
    });
  }

  function setStatusBadge(status) {
    const el = document.getElementById('status-badge');
    if (!el) return;
    const labels = {
      idle: 'Ожидание',
      translating: 'Перевод',
      generating: 'Генерация',
      stitching: 'Сшивка',
      done: 'Готово',
      error: 'Ошибка'
    };
    el.textContent = labels[status] || status;
    el.className = 'badge ';
    if (status === 'done') el.classList.add('bg-success');
    else if (status === 'error') el.classList.add('bg-danger');
    else if (status === 'idle') el.classList.add('bg-secondary');
    else el.classList.add('bg-primary');
  }

  function setProgress(percent) {
    const bar = document.getElementById('progress-bar');
    if (!bar) return;
    bar.style.width = percent + '%';
    bar.setAttribute('aria-valuenow', percent);
    bar.textContent = percent + '%';
  }

  function setLog(lines) {
    const area = document.getElementById('log-area');
    if (!area) return;
    area.textContent = Array.isArray(lines) ? lines.join('\n') : '';
    area.scrollTop = area.scrollHeight;
  }

  function showResult(videoUrl) {
    const card = document.getElementById('result-card');
    const video = document.getElementById('result-video');
    const downloadBtn = document.getElementById('btn-download');
    if (!card || !video || !downloadBtn) return;
    card.style.display = 'block';
    video.src = videoUrl;
    downloadBtn.href = videoUrl;
    downloadBtn.download = 'video.mp4';
  }

  function hideResult() {
    const card = document.getElementById('result-card');
    const video = document.getElementById('result-video');
    if (card) card.style.display = 'none';
    if (video) video.src = '';
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  function poll() {
    if (!currentTaskId) return;
    api('/api/status?task_id=' + encodeURIComponent(currentTaskId)).then(function (data) {
      setStatusBadge(data.status);
      setProgress(data.progress || 0);
      if (data.status === 'done' && data.video_url) {
        showResult(data.video_url);
        stopPolling();
        currentTaskId = null;
      } else if (data.status === 'error') {
        stopPolling();
        currentTaskId = null;
      }
    }).catch(function () {});
    api('/api/log?task_id=' + encodeURIComponent(currentTaskId)).then(function (data) {
      setLog(data.lines || []);
    }).catch(function () {});
  }

  document.getElementById('btn-generate').addEventListener('click', function () {
    const promptEl = document.getElementById('prompt-input');
    const modelEl = document.getElementById('model-select');
    const prompt = (promptEl && promptEl.value || '').trim();
    const modelId = (modelEl && modelEl.value) || (modelEl && modelEl.options.length && modelEl.options[0].value) || '';
    if (!prompt) {
      alert('Введите промпт.');
      return;
    }
    hideResult();
    stopPolling();
    setStatusBadge('idle');
    setProgress(0);
    setLog([]);
    api('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: prompt, model_id: modelId })
    }).then(function (data) {
      currentTaskId = data.task_id;
      poll();
      pollInterval = setInterval(poll, POLL_MS);
    }).catch(function (err) {
      alert(err.message || 'Ошибка запуска генерации');
    });
  });

  document.getElementById('model-select').addEventListener('change', function () {
    var opt = this.options[this.selectedIndex];
    var desc = document.getElementById('model-description');
    if (desc && opt) desc.textContent = opt.title || opt.text || '';
  });

  var modelSelect = document.getElementById('model-select');
  if (modelSelect && modelSelect.options[modelSelect.selectedIndex]) {
    document.getElementById('model-description').textContent = modelSelect.options[modelSelect.selectedIndex].title || modelSelect.options[modelSelect.selectedIndex].text || '';
  }

  api('/api/models').then(function (models) {
    var sel = document.getElementById('model-select');
    if (!sel || !Array.isArray(models)) return;
    sel.innerHTML = '';
    models.forEach(function (m) {
      var opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name;
      opt.title = m.description || '';
      if (m.insufficient_vram) opt.classList.add('text-danger');
      sel.appendChild(opt);
    });
    var desc = document.getElementById('model-description');
    if (sel.options.length && desc) desc.textContent = sel.options[0].title || sel.options[0].text || '—';
  }).catch(function () {});
})();
