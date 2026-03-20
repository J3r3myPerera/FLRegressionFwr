/* ================================================================
   GLOBALS
   ================================================================ */
const COLORS = {
  FedAvg: "#e74c3c",
  FedProx: "#3498db",
  SmartFedProx: "#2ecc71",
};
const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e0e0e8", size: 12 },
  margin: { t: 40, b: 50, l: 60, r: 20 },
  xaxis: { gridcolor: "#2d2f3a" },
  yaxis: { gridcolor: "#2d2f3a" },
  legend: { orientation: "h", y: -0.2 },
  hovermode: "x unified",
};

/* ================================================================
   ON LOAD — fetch server config & populate strategy cards
   ================================================================ */
(async function init() {
  try {
    const resp = await fetch("/api/config");
    const cfg = resp.ok ? await resp.json() : null;
    if (!cfg) return;

    // Show server config in sidebar (populate form defaults from server)
    document.getElementById("numClients").value = cfg.num_clients;
    document.getElementById("fractionFit").value = cfg.fraction_fit;
    document.getElementById("localEpochs").value = cfg.local_epochs;

    const el = document.getElementById("runtimeConfig");
    el.innerHTML = `<strong>Device:</strong> ${cfg.device} &middot; <strong>Input dim:</strong> ${cfg.input_dim}`;

    // Strategy info cards
    const container = document.getElementById("strategyCards");
    for (const [name, info] of Object.entries(cfg.strategies)) {
      const card = document.createElement("div");
      card.className = "strategy-card";
      card.innerHTML = `<h4>${name}</h4>
        <p>&mu; = ${info.proximal_mu}${info.adaptive_mu_enabled ? " (adaptive)" : ""}<br>
        Selection: ${info.selection_strategy}<br>
        ${info.description}</p>`;
      container.appendChild(card);
    }
  } catch (_) {
    /* API not ready yet */
  }
})();

/* ================================================================
   RUN SIMULATION
   ================================================================ */
async function runSimulation() {
  const strategies = [];
  if (document.getElementById("chkFedAvg").checked)
    strategies.push("FedAvg");
  if (document.getElementById("chkFedProx").checked)
    strategies.push("FedProx");
  if (document.getElementById("chkSmartFedProx").checked)
    strategies.push("SmartFedProx");
  const validationError = document.getElementById("validationError");
  if (strategies.length === 0) {
    validationError.textContent = "Select at least one strategy.";
    validationError.classList.remove("hidden");
    return;
  }
  validationError.classList.add("hidden");

  const payload = {
    strategies,
    num_rounds: parseInt(document.getElementById("numRounds").value),
    num_trials: parseInt(document.getElementById("numTrials").value),
    seed: parseInt(document.getElementById("seed").value),
    num_clients: parseInt(document.getElementById("numClients").value),
    fraction_fit: parseFloat(
      document.getElementById("fractionFit").value,
    ),
    local_epochs: parseInt(document.getElementById("localEpochs").value),
  };

  document.getElementById("btnRun").disabled = true;
  document.getElementById("progressSection").classList.remove("hidden");
  document.getElementById("progressBar").style.width = "30%";
  document.getElementById("statusText").innerHTML =
    '<span class="spinner"></span> Simulating&hellip; This may take a few minutes.';

  try {
    const resp = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    document.getElementById("progressBar").style.width = "100%";

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Simulation failed");
    }

    const data = await resp.json();
    renderResults(data.result);
  } catch (e) {
    document.getElementById("statusText").textContent =
      "Error: " + e.message;
  } finally {
    document.getElementById("btnRun").disabled = false;
  }
}

/* ================================================================
   RESET
   ================================================================ */
function resetUI() {
  document.getElementById("welcomeScreen").classList.remove("hidden");
  document.getElementById("resultsScreen").classList.add("hidden");
  document.getElementById("progressSection").classList.add("hidden");
  document.getElementById("progressBar").style.width = "0%";
}

/* ================================================================
   RENDER RESULTS
   ================================================================ */
function renderResults(result) {
  document.getElementById("welcomeScreen").classList.add("hidden");
  document.getElementById("resultsScreen").classList.remove("hidden");
  document.getElementById("progressSection").classList.add("hidden");

  const { config, metrics, summary, winner } = result;

  /* ---- summary metric boxes ---- */
  const metricsRow = document.getElementById("summaryMetrics");
  metricsRow.innerHTML = [
    { label: "Strategies", value: config.strategies.length },
    { label: "Rounds", value: config.num_rounds },
    { label: "Clients", value: config.num_clients },
    { label: "Trials", value: config.num_trials },
  ]
    .map(
      (m) =>
        `<div class="metric-box"><div class="value">${m.value}</div><div class="label">${m.label}</div></div>`,
    )
    .join("");

  /* ---- winner ---- */
  const winnerR2 = metrics[winner].r2_scores.at(-1);
  const comparisons = Object.keys(metrics)
    .filter((n) => n !== winner)
    .map((n) => {
      const otherR2 = metrics[n].r2_scores.at(-1);
      const pct = ((winnerR2 - otherR2) / Math.abs(otherR2)) * 100;
      return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}% vs ${n}`;
    })
    .join(" &nbsp;·&nbsp; ");
  document.getElementById("winnerCard").innerHTML =
    `Best performing strategy: <strong>${winner}</strong> (Final R² = ${winnerR2.toFixed(4)})` +
    (comparisons ? `<div style="margin-top:0.4rem;font-size:0.82rem;color:var(--muted)">${comparisons}</div>` : "");

  /* ---- comparison table ---- */
  const tHead = document.querySelector("#comparisonTable thead");
  const tBody = document.querySelector("#comparisonTable tbody");
  tHead.innerHTML =
    "<tr><th>Strategy</th><th>Final R²</th><th>Best R²</th><th>Final MSE</th><th>Lowest MSE</th><th>Final &mu;</th></tr>";
  tBody.innerHTML = summary
    .map(
      (s) =>
        `<tr>
      <td>${s.strategy}${s.strategy === winner ? '<span class="winner-badge">BEST</span>' : ""}</td>
      <td>${s.final_r2}</td><td>${s.best_r2}</td>
      <td>${s.final_mse}</td><td>${s.lowest_mse}</td><td>${s.final_mu}</td>
    </tr>`,
    )
    .join("");

  /* ---- charts ---- */
  const traces = (key) =>
    Object.entries(metrics).map(([name, m]) => ({
      x: m.rounds,
      y: m[key],
      mode: "lines+markers",
      name,
      line: { color: COLORS[name], width: 2 },
      marker: { size: 6 },
    }));

  Plotly.newPlot(
    "chartR2",
    traces("r2_scores"),
    {
      ...PLOT_LAYOUT,
      title: "R² Score",
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "R²" },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "Round" },
    },
    { responsive: true },
  );
  Plotly.newPlot(
    "chartMSE",
    traces("mse_losses"),
    {
      ...PLOT_LAYOUT,
      title: "MSE Loss",
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "MSE" },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "Round" },
    },
    { responsive: true },
  );
  Plotly.newPlot(
    "chartTrainLoss",
    traces("avg_train_loss"),
    {
      ...PLOT_LAYOUT,
      title: "Avg Training Loss",
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "Loss" },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "Round" },
    },
    { responsive: true },
  );
  Plotly.newPlot(
    "chartDivergence",
    traces("avg_divergence"),
    {
      ...PLOT_LAYOUT,
      title: "Avg Model Divergence",
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "Divergence" },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "Round" },
    },
    { responsive: true },
  );
  Plotly.newPlot(
    "chartMu",
    traces("avg_effective_mu"),
    {
      ...PLOT_LAYOUT,
      title: "Avg Effective μ",
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "μ" },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "Round" },
    },
    { responsive: true },
  );

  // Bar chart
  const names = Object.keys(metrics);
  Plotly.newPlot(
    "chartBar",
    [
      {
        x: names,
        y: names.map((n) => metrics[n].r2_scores.at(-1)),
        type: "bar",
        name: "Final R²",
        marker: { color: names.map((n) => COLORS[n]) },
        text: names.map((n) => metrics[n].r2_scores.at(-1).toFixed(4)),
        textposition: "outside",
      },
    ],
    {
      ...PLOT_LAYOUT,
      title: "Final R² Comparison",
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "R²" },
    },
    { responsive: true },
  );

  /* ---- detailed round table ---- */
  const dHead = document.querySelector("#detailedTable thead");
  const dBody = document.querySelector("#detailedTable tbody");
  dHead.innerHTML =
    "<tr><th>Round</th><th>Strategy</th><th>R²</th><th>MSE</th><th>Train Loss</th><th>Divergence</th><th>&mu;</th></tr>";
  let csvRows = ["Round,Strategy,R2,MSE,TrainLoss,Divergence,Mu"];
  let rows = "";
  for (const [name, m] of Object.entries(metrics)) {
    m.rounds.forEach((r, i) => {
      const vals = [
        r,
        name,
        m.r2_scores[i].toFixed(4),
        m.mse_losses[i].toFixed(4),
        m.avg_train_loss[i].toFixed(4),
        m.avg_divergence[i].toFixed(4),
        m.avg_effective_mu[i].toFixed(4),
      ];
      rows += `<tr>${vals.map((v) => `<td>${v}</td>`).join("")}</tr>`;
      csvRows.push(vals.join(","));
    });
  }
  dBody.innerHTML = rows;

  // CSV download
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const dl = document.getElementById("downloadCSV");
  dl.href = url;
  dl.download = `fl_results_${Date.now()}.csv`;
}
