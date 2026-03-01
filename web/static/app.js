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

  const LOG_POLL_INTERVAL_MS = 3000;
  const STATUS_POLL_INTERVAL_MS = 2000;
  let statusPollTimer = null;

  function showJobStatus(jobId) {
    jobStatus.classList.remove("d-none");
    jobIdEl.textContent = jobId;
    statusTextEl.textContent = "ожидание…";
    downloadWrap.classList.add("d-none");
    errorWrap.classList.add("d-none");
    errorWrap.textContent = "";
  }

  function updateStatus(data) {
    statusTextEl.textContent = data.status;
    if (data.status === "done" && data.download_url) {
      downloadWrap.classList.remove("d-none");
      downloadLink.href = data.download_url;
      if (statusPollTimer) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
      }
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
    fetch("/api/logs?tail=200")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        logContainer.textContent = (data.lines && data.lines.length)
          ? data.lines.join("\n")
          : "(логов пока нет)";
        logContainer.scrollTop = logContainer.scrollHeight;
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
