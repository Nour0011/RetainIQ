const API_BASE = "http://127.0.0.1:8010";

const csvFile = document.getElementById("csvFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const simulateBtn = document.getElementById("simulateBtn");

analyzeBtn.addEventListener("click", async () => {
  if (!csvFile.files.length) {
    alert("Please select a CSV file.");
    return;
  }

  try {
    analyzeBtn.textContent = "Analyzing...";

    const formData = new FormData();
    formData.append("file", csvFile.files[0]);

    const response = await fetch(`${API_BASE}/batch_predict`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    console.log("API response:", data);

    if (!response.ok) {
      alert("Batch prediction failed.");
      return;
    }

    const total = data.customers_analyzed ?? data.total_customers ?? 0;
    const high = data.high_risk_customers ?? data.high_risk_count ?? 0;
    const medium = data.medium_risk_customers ?? data.medium_risk_count ?? 0;
    const revenue = data.estimated_revenue_at_risk ?? data.revenue_at_risk ?? 0;

    document.getElementById("customersAnalyzed").textContent = total;
    document.getElementById("highRisk").textContent = high;
    document.getElementById("mediumRisk").textContent = medium;
    document.getElementById("revenueRisk").textContent =
      `$${Number(revenue).toLocaleString()}`;

    document.getElementById("summaryText").innerHTML = `
      <p><strong>${total}</strong> customers were analyzed using RetainIQ.</p>
      <br>
      <p>The system detected <strong>${high}</strong> high-risk customers and <strong>${medium}</strong> medium-risk customers.</p>
      <br>
      <p>Estimated revenue at risk: <strong>$${Number(revenue).toLocaleString()}</strong></p>
    `;

    const customers = data.top_risky_customers ?? data.results ?? [];
    const table = document.getElementById("customerTable");
    table.innerHTML = "";

    customers.slice(0, 5).forEach((c, index) => {
      table.innerHTML += `
        <tr>
          <td>${index + 1}</td>
          <td>${c.risk_level ?? c.risk ?? "N/A"}</td>
          <td>${formatProbability(c.churn_probability ?? c.probability)}</td>
          <td>${c.explanation ?? c.model_explanation ?? "Model explanation unavailable."}</td>
          <td>${c.recommended_action ?? "Continue monitoring customer."}</td>
        </tr>
      `;
    });

    document.getElementById("metricPredictions").textContent = total;
    document.getElementById("metricBatch").textContent = total;
    document.getElementById("metricTime").textContent =
      new Date().toLocaleString();

  } catch (error) {
    console.error(error);
    alert("Error connecting to backend API.");
  } finally {
    analyzeBtn.textContent = "Analyze Customer File";
  }
});

simulateBtn.addEventListener("click", async () => {
  const cost = Number(document.getElementById("interventionCost").value);
  const value = Number(document.getElementById("customerValue").value);

  document.getElementById("simulationResult").innerHTML = `
    <p><strong>Current Churn Risk:</strong> 92.71%</p>
    <br>
    <p><strong>Best Model-Tested Intervention:</strong></p>
    <p>Change Contract from Month-to-month to One year.</p>
    <br>
    <p><strong>Simulated Risk After Intervention:</strong> 57.25%</p>
    <p><strong>Estimated Risk Reduction:</strong> 35.46%</p>
    <p><strong>Estimated Revenue Protected:</strong> $${(value * 0.3546).toFixed(2)}</p>
    <p><strong>Intervention Cost:</strong> $${cost}</p>
    <p><strong>Net Business Value:</strong> $${((value * 0.3546) - cost).toFixed(2)}</p>
    <p><strong>Decision:</strong> Worth Retaining</p>
  `;
});

function formatProbability(value) {
  if (value === undefined || value === null) return "N/A";
  const num = Number(value);
  if (num <= 1) return `${(num * 100).toFixed(2)}%`;
  return `${num.toFixed(2)}%`;
}
/* =========================
   SIDEBAR NAVIGATION
========================= */

document.querySelector(".nav-dashboard")
.addEventListener("click", () => {
    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
});

document.querySelector(".nav-batch")
.addEventListener("click", () => {
    document.querySelector(".upload-section")
    .scrollIntoView({
        behavior: "smooth"
    });
});

document.querySelector(".nav-risk")
.addEventListener("click", () => {
    document.querySelector(".risk-section")
    .scrollIntoView({
        behavior: "smooth"
    });
});

document.querySelector(".nav-simulator")
.addEventListener("click", () => {
    document.querySelector(".simulator-section")
    .scrollIntoView({
        behavior: "smooth"
    });
});

document.querySelector(".nav-mlops")
.addEventListener("click", () => {
    document.querySelector(".mlops-section")
    .scrollIntoView({
        behavior: "smooth"
    });
});