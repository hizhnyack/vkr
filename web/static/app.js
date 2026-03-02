(function () {
  "use strict";

  const form = document.getElementById("run-form");
  const videoInput = document.getElementById("video");
  const taskInput = document.getElementById("task");
  const submitBtn = document.getElementById("submit-btn");
  const jobStatus = document.getElementById("job-status");
  const jobIdEl = document.getElementById("job-id");
  const statusTextEl = document.getElementById("status-text");
  const downloadWrap = document.getElementById("download-wrap");
  const downloadLink = document.getElementById("download-link");
  const errorWrap = document.getElementById("error-wrap");
  const logContainer = document.getElementById("log-container");
  const refreshLogsBtn = document.getElementById("refresh-logs");
  const pipelineStageIdle = document.getElementById("pipeline-stage-idle");
  const pipelineStageList = document.getElementById("pipeline-stage-list");
  const pipelineStageUnknown = document.getElementById("pipeline-stage-unknown");
  const reportWrap = document.getElementById("report-wrap");

  const PIPELINE_STAGES = [
    "Метаданные и кадры (ffprobe/decord)",
    "LLM: текст → JSON сценария",
    "Сэмпл кадров",
    "VLM: время суток, погода",
    "Depth: карта глубины",
    "Планирование вставок",
    "Генерация патчей (SD + ControlNet)",
    "Композитинг",
    "Запись видео (ffmpeg)",
    "Разметка (YOLO)"
  ];

  function renderPipelineStages(currentStageText, allDone) {
    var currentIndex = (!allDone && currentStageText) ? PIPELINE_STAGES.indexOf(currentStageText) : -1;
    if (allDone) currentIndex = PIPELINE_STAGES.length;
    pipelineStageIdle.classList.add("d-none");
    pipelineStageUnknown.classList.add("d-none");
    pipelineStageList.classList.remove("d-none");
    pipelineStageList.innerHTML = "";
    PIPELINE_STAGES.forEach(function (label, i) {
      var li = document.createElement("li");
      li.className = "list-group-item list-group-item-action py-1 small";
      li.textContent = label;
      if (i < currentIndex) {
        li.classList.add("list-group-item-success");
      } else if (i === currentIndex && !allDone) {
        li.classList.add("list-group-item-primary", "fw-bold");
      }
      pipelineStageList.appendChild(li);
    });
    if (!allDone && currentStageText && currentIndex === -1) {
      pipelineStageUnknown.textContent = "Текущий: " + currentStageText;
      pipelineStageUnknown.classList.remove("d-none");
    }
  }

  function showJobStatus(jobId) {
    jobStatus.classList.remove("d-none");
    jobIdEl.textContent = jobId;
    statusTextEl.textContent = "ожидание…";
    downloadWrap.classList.add("d-none");
    errorWrap.classList.add("d-none");
    errorWrap.textContent = "";
    reportWrap.classList.add("d-none");
    reportWrap.innerHTML = "";
    pipelineStageIdle.classList.add("d-none");
    pipelineStageList.classList.remove("d-none");
    renderPipelineStages(null);
  }

  function updateStatus(data) {
    statusTextEl.textContent = data.status;
    if (data.status === "done") {
      renderPipelineStages(null, true);
    } else if (data.stage !== undefined) {
      renderPipelineStages(data.stage);
    }
    if (data.status === "done" && data.download_url) {
      downloadWrap.classList.remove("d-none");
      downloadLink.href = data.download_url;
      if (statusPollTimer) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
      }
      loadReport(data.job_id);
    }
    if (data.status === "error" && data.error) {
      errorWrap.classList.remove("d-none");
      errorWrap.textContent = data.error;
      if (statusPollTimer) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
      }
    }
  }

  function loadReport(jobId) {
    fetch("/api/report/" + encodeURIComponent(jobId))
      .then(function (r) {
        if (!r.ok) return;
        return r.json();
      })
      .then(function (report) {
        if (!report) return;
        reportWrap.classList.remove("d-none");
        var summary = report.summary || "";
        var events = report.events || [];
        var html = "<p class=\"fw-bold mb-1\">Что добавлено на видео</p>";
        if (summary) html += "<p class=\"small mb-2\">" + summary + "</p>";
        if (events.length) {
          html += "<ul class=\"small mb-0\">";
          events.forEach(function (ev) {
            var line = ev.object_type + " (кадры " + ev.start_frame + "–" + ev.end_frame;
            if (ev.start_sec !== undefined && ev.end_sec !== undefined) {
              line += ", " + ev.start_sec.toFixed(1) + "–" + ev.end_sec.toFixed(1) + " с";
            }
            line += ")";
            if (ev.description) line += " — " + ev.description;
            html += "<li>" + line + "</li>";
          });
          html += "</ul>";
        }
        reportWrap.innerHTML = html;
      })
      .catch(function () {});
  }

  function pollStatus(jobId) {
    if (statusPollTimer) return;
    function tick() {
      fetch("/api/status/" + encodeURIComponent(jobId))
        .then(function (r) {
          if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || r.status); });
          return r.json();
        })
        .then(updateStatus)
        .catch(function (err) {
          statusTextEl.textContent = "ошибка запроса";
          errorWrap.classList.remove("d-none");
          errorWrap.textContent = err.message;
          if (statusPollTimer) {
            clearInterval(statusPollTimer);
            statusPollTimer = null;
          }
        });
    }
    tick();
    statusPollTimer = setInterval(tick, STATUS_POLL_INTERVAL_MS);
  }

  function loadLogs() {
    var wasAtBottom = (logContainer.scrollHeight - logContainer.scrollTop - logContainer.clientHeight) < 20;
    fetch("/api/logs?tail=500")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        logContainer.textContent = (data.lines && data.lines.length)
          ? data.lines.join("\n")
          : "(логов пока нет)";
        var hasSelection = window.getSelection().toString().length > 0;
        if (wasAtBottom && !hasSelection) {
          logContainer.scrollTop = logContainer.scrollHeight;
        }
      })
      .catch(function () {
        logContainer.textContent = "Не удалось загрузить логи.";
      });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (!videoInput.files.length) return;
    var fd = new FormData();
    fd.append("video", videoInput.files[0]);
    fd.append("task", taskInput.value.trim());
    submitBtn.disabled = true;
    fetch("/run", {
      method: "POST",
      body: fd,
    })
      .then(function (r) {
        return r.json().then(function (data) {
          if (!r.ok) throw new Error(data.error || "Ошибка запроса");
          return data;
        });
      })
      .then(function (data) {
        showJobStatus(data.job_id);
        pollStatus(data.job_id);
        loadLogs();
      })
      .catch(function (err) {
        alert(err.message);
      })
      .finally(function () {
        submitBtn.disabled = false;
      });
  });

  refreshLogsBtn.addEventListener("click", loadLogs);

  setInterval(loadLogs, LOG_POLL_INTERVAL_MS);
  loadLogs();
})();
