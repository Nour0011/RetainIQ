// ======================================================
// RetainIQ Frontend Logic
// ======================================================

// Backend API URL
const API_BASE = "http://127.0.0.1:8010";

// Main page elements
const csvFile = document.getElementById("csvFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const simulateBtn = document.getElementById("simulateBtn");

// Store analyzed customers globally so the simulator can use them
let latestRiskyCustomers = [];

// ======================================================
// Batch Customer Analysis
// ======================================================

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

    if (!response.ok) {
      alert("Batch prediction failed.");
      return;
    }

    const total = data.customers_analyzed ?? 0;
    const high = data.high_risk_customers ?? 0;
    const medium = data.medium_risk_customers ?? 0;
    const revenue = data.estimated_revenue_at_risk ?? 0;

    document.getElementById("customersAnalyzed").textContent = total;
    document.getElementById("highRisk").textContent = high;
    document.getElementById("mediumRisk").textContent = medium;

    document.getElementById("revenueRisk").textContent =
      `$${Number(revenue).toLocaleString()}`;

    document.getElementById("summaryText").innerHTML = `
      <p><strong>${total}</strong> customers were analyzed.</p>
      <br>
      <p>
        RetainIQ detected <strong>${high}</strong> high-risk customers and
        <strong>${medium}</strong> medium-risk customers.
      </p>
      <br>
      <p>
        Estimated revenue at risk:
        <strong>$${Number(revenue).toLocaleString()}</strong>
      </p>
    `;

    latestRiskyCustomers = data.top_risky_customers ?? [];

    renderCustomerTable(latestRiskyCustomers);

    document.getElementById("metricPredictions").textContent = total;
    document.getElementById("metricBatch").textContent = total;
    document.getElementById("metricTime").textContent =
      new Date().toLocaleString();

    document.getElementById("simulationResult").innerHTML = `
      <p>
        Customer analysis completed. Click
        <strong>Simulate Best Retention Strategy</strong>
        to test the recommended action for the highest-risk customer.
      </p>
    `;

  } catch (error) {
    console.error(error);
    alert("Error connecting to backend API.");
  } finally {
    analyzeBtn.textContent = "Analyze Customer File";
  }
});

// ======================================================
// Render High-Risk Customer Table
// ======================================================

function renderCustomerTable(customers) {
  const table = document.getElementById("customerTable");
  table.innerHTML = "";

  if (!customers.length) {
    table.innerHTML = `
      <tr>
        <td colspan="6">No customer results available.</td>
      </tr>
    `;
    return;
  }

  customers.slice(0, 5).forEach((customer, index) => {
    table.innerHTML += `
      <tr>
        <td>${index + 1}</td>
        <td>${customer.customer_id ?? `Demo Customer ${index + 1}`}</td>
        <td>${customer.risk_level ?? "N/A"}</td>
        <td>${formatProbability(customer.churn_probability)}</td>
        <td>${customer.explanation ?? "Model explanation unavailable."}</td>
        <td>${customer.recommended_action ?? "Continue monitoring customer."}</td>
      </tr>
    `;
  });
}

// ======================================================
// AI What-If Retention Simulator
// ======================================================

simulateBtn.addEventListener("click", () => {
  if (!latestRiskyCustomers.length) {
    alert("Please upload and analyze a CSV file first.");
    return;
  }

  // Select highest-risk customer
  const selectedCustomer = latestRiskyCustomers[0];

  const customerId =
    selectedCustomer.customer_id ??
    "Highest-Risk Demo Customer";

  const currentRisk =
    Number(selectedCustomer.churn_probability ?? 0) * 100;

  const explanation =
    selectedCustomer.explanation ??
    "The model found a possible retention action.";

  const action =
    selectedCustomer.recommended_action ??
    "Apply recommended retention action.";

  // Extract risk reduction from explanation if possible
  const reductionMatch = explanation.match(/reduces predicted churn risk by ([\d.]+)%/);

  const riskReduction = reductionMatch
    ? Number(reductionMatch[1])
    : 25.0;

  const afterRisk = Math.max(currentRisk - riskReduction, 0);

  document.getElementById("simulationResult").innerHTML = `
    <p><strong>Selected Customer:</strong> ${customerId}</p>

    <br>

    <p><strong>Current Churn Risk:</strong> ${currentRisk.toFixed(2)}%</p>

    <br>

    <p><strong>AI Recommended Retention Action:</strong></p>
    <p>${action}</p>

    <br>

    <p><strong>What-If Scenario:</strong></p>
    <p>
      RetainIQ tests what may happen if this recommended action is applied
      to the selected customer.
    </p>

    <br>

    <p><strong>Predicted Churn Risk After Action:</strong> ${afterRisk.toFixed(2)}%</p>

    <p><strong>Estimated Risk Reduction:</strong> ${riskReduction.toFixed(2)}%</p>

    <br>

    <p><strong>Business Meaning:</strong></p>
    <p>
      The customer is still risky, but the recommended action may significantly
      reduce churn probability. This helps the company decide which customer
      should receive retention attention first.
    </p>
  `;
});

// ======================================================
// Probability Formatter
// ======================================================

function formatProbability(value) {
  if (value === undefined || value === null) {
    return "N/A";
  }

  const num = Number(value);

  if (num <= 1) {
    return `${(num * 100).toFixed(2)}%`;
  }

  return `${num.toFixed(2)}%`;
}

// ======================================================
// Sidebar Navigation
// ======================================================

document.querySelector(".nav-dashboard").addEventListener("click", () => {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
});

document.querySelector(".nav-batch").addEventListener("click", () => {
  document.querySelector(".upload-section").scrollIntoView({
    behavior: "smooth"
  });
});

document.querySelector(".nav-risk").addEventListener("click", () => {
  document.querySelector(".risk-section").scrollIntoView({
    behavior: "smooth"
  });
});

document.querySelector(".nav-simulator").addEventListener("click", () => {
  document.querySelector(".simulator-section").scrollIntoView({
    behavior: "smooth"
  });
});

document.querySelector(".nav-mlops").addEventListener("click", () => {
  document.querySelector(".mlops-section").scrollIntoView({
    behavior: "smooth"
  });
});