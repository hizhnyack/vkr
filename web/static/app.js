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

  const LOG_POLL_INTERVAL_MS = 3000;
  const STATUS_POLL_INTERVAL_MS = 2000;
  let statusPollTimer = null;

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

  var pipelineMaxStageIndex = -1;

  function renderPipelineStages(currentStageText, allDone) {
    pipelineStageUnknown.classList.add("d-none");
    if (allDone) {
      pipelineMaxStageIndex = PIPELINE_STAGES.length - 1;
    } else if (currentStageText) {
      var idx = PIPELINE_STAGES.indexOf(currentStageText);
      if (idx !== -1 && idx > pipelineMaxStageIndex) {
        pipelineMaxStageIndex = idx;
      }
    }
    if (pipelineMaxStageIndex < 0) {
      pipelineStageIdle.classList.remove("d-none");
      pipelineStageList.classList.add("d-none");
      pipelineStageList.innerHTML = "";
      return;
    }
    pipelineStageIdle.classList.add("d-none");
    pipelineStageList.classList.remove("d-none");
    pipelineStageList.innerHTML = "";
    pipelineStageList.style.counterReset = "step 0";
    for (var i = 0; i <= pipelineMaxStageIndex && i < PIPELINE_STAGES.length; i++) {
      var li = document.createElement("li");
      li.className = "pipeline-step";
      li.textContent = PIPELINE_STAGES[i];
      if (allDone || i < pipelineMaxStageIndex) {
        li.classList.add("step-done");
      } else {
        li.classList.add("step-current");
      }
      pipelineStageList.appendChild(li);
    }
    if (!allDone && currentStageText && PIPELINE_STAGES.indexOf(currentStageText) === -1) {
      pipelineStageUnknown.textContent = "Текущий: " + currentStageText;
      pipelineStageUnknown.classList.remove("d-none");
    }
  }

  function showJobStatus(jobId) {
    jobStatus.classList.remove("d-none");
    jobIdEl.textContent = jobId;
    statusTextEl.textContent = "ожидание…";
    downloadWrap.classList.add("d-none");
    downloadLink.removeAttribute("href");
    errorWrap.classList.add("d-none");
    errorWrap.textContent = "";
    reportWrap.classList.add("d-none");
    reportWrap.innerHTML = "";
    pipelineMaxStageIndex = -1;
    pipelineStageIdle.classList.remove("d-none");
    pipelineStageList.classList.add("d-none");
    pipelineStageList.innerHTML = "";
  }

  function updateStatus(data) {
    statusTextEl.textContent = data.status;
    if (data.status === "done") {
      renderPipelineStages(null, true);
    } else if (data.stage !== undefined) {
      renderPipelineStages(data.stage);
    }
    if (data.status === "done" && data.job_id) {
      downloadWrap.classList.remove("d-none");
      var downloadUrl = data.download_url || (window.location.pathname.replace(/\/?$/, "") + "/download/" + encodeURIComponent(data.job_id));
      downloadLink.href = downloadUrl;
      downloadLink.setAttribute("download", "result_" + data.job_id + ".mp4");
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
