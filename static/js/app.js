/**
 * Explainable Credit Underwriting & Default Scoring Engine - Client Logic
 * Handles real-time input synchronization, dynamic SHAP visualizer, 
 * API communication, and interactive audit logging.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Form Inputs & Displays
  const ageSlider = document.getElementById("age-slider");
  const ageInput = document.getElementById("age-input");
  const ageDisplay = document.getElementById("age-val-display");
  const ageError = document.getElementById("age-error");

  const incomeSlider = document.getElementById("income-slider");
  const incomeInput = document.getElementById("income-input");
  const incomeDisplay = document.getElementById("income-val-display");
  const incomeError = document.getElementById("income-error");

  const loanSlider = document.getElementById("loan-slider");
  const loanInput = document.getElementById("loan-input");
  const loanDisplay = document.getElementById("loan-val-display");
  const loanError = document.getElementById("loan-error");

  const scoreSlider = document.getElementById("credit-score-slider");
  const scoreInput = document.getElementById("credit-score-input");
  const scoreDisplay = document.getElementById("score-val-display");
  const scoreError = document.getElementById("score-error");

  const historySlider = document.getElementById("default-history-slider");
  const historyInput = document.getElementById("default-history-input");
  const historyDisplay = document.getElementById("history-val-display");
  const historyError = document.getElementById("history-error");

  const dtiDisplay = document.getElementById("dti-ratio-display");
  const creditTierDisplay = document.getElementById("credit-tier-display");
  const underwritingForm = document.getElementById("underwriting-form");
  const submitBtn = document.getElementById("submit-btn");

  // DOM Elements - Assessment Visuals
  const gaugeCircle = document.getElementById("gauge-circle");
  const probPercentage = document.getElementById("prob-percentage");
  const riskTierPill = document.getElementById("risk-tier-pill");
  const decisionActionTitle = document.getElementById("decision-action-title");
  const decisionRationale = document.getElementById("decision-rationale");
  const shapContainer = document.getElementById("shap-factors-container");
  const auditIdBadge = document.getElementById("audit-id-badge");
  const auditTableBody = document.getElementById("audit-table-body");

  // Modal Elements
  const metricsModal = document.getElementById("metrics-modal");
  const openMetricsBtn = document.getElementById("open-metrics-modal-btn");
  const closeModalBtn = document.getElementById("close-modal-btn");

  // State
  let sessionAuditRecords = [];

  // Circumference of radial gauge (r = 70) => 2 * pi * 70 = 439.82
  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 70;
  gaugeCircle.style.strokeDasharray = GAUGE_CIRCUMFERENCE;
  gaugeCircle.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;

  // -------------------------------------------------------------------------
  // Helper: Format Currency
  // -------------------------------------------------------------------------
  function formatCurrency(val) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(val);
  }

  // -------------------------------------------------------------------------
  // Live Input Sync and Financial Calculations
  // -------------------------------------------------------------------------
  function updateRatios() {
    const income = parseFloat(incomeInput.value) || 1;
    const loan = parseFloat(loanInput.value) || 0;
    const score = parseInt(scoreInput.value) || 300;

    // DTI Ratio
    const dti = loan / Math.max(income, 1);
    let dtiStatus = "Moderate";
    if (dti < 2.5) dtiStatus = "Low";
    else if (dti > 4.5) dtiStatus = "High";
    dtiDisplay.textContent = `${dti.toFixed(2)}x (${dtiStatus})`;

    // Credit Bureau Tier
    let tier = "Subprime (<620)";
    if (score >= 800) tier = "Super Prime (800+)";
    else if (score >= 740) tier = "Prime (740-799)";
    else if (score >= 670) tier = "Near Prime (670-739)";
    else if (score >= 580) tier = "Fair (580-669)";
    creditTierDisplay.textContent = tier;
  }

  function bindInputSync(slider, input, display, formatter) {
    slider.addEventListener("input", (e) => {
      input.value = e.target.value;
      display.textContent = formatter(e.target.value);
      updateRatios();
    });

    input.addEventListener("input", (e) => {
      slider.value = e.target.value;
      display.textContent = formatter(e.target.value);
      updateRatios();
    });
  }

  bindInputSync(ageSlider, ageInput, ageDisplay, (v) => `${v} yrs`);
  bindInputSync(incomeSlider, incomeInput, incomeDisplay, (v) => formatCurrency(v));
  bindInputSync(loanSlider, loanInput, loanDisplay, (v) => formatCurrency(v));
  bindInputSync(scoreSlider, scoreInput, scoreDisplay, (v) => `${v}`);
  bindInputSync(historySlider, historyInput, historyDisplay, (v) => `${v} ${v == 1 ? "default" : "defaults"}`);

  updateRatios();

  // -------------------------------------------------------------------------
  // Archetype Loaders
  // -------------------------------------------------------------------------
  function setFormValues(age, income, loan, score, history, autoSubmit = true) {
    ageSlider.value = ageInput.value = age;
    ageDisplay.textContent = `${age} yrs`;

    incomeSlider.value = incomeInput.value = income;
    incomeDisplay.textContent = formatCurrency(income);

    loanSlider.value = loanInput.value = loan;
    loanDisplay.textContent = formatCurrency(loan);

    scoreSlider.value = scoreInput.value = score;
    scoreDisplay.textContent = `${score}`;

    historySlider.value = historyInput.value = history;
    historyDisplay.textContent = `${history} defaults`;

    updateRatios();

    if (autoSubmit) {
      handleFormSubmit();
    }
  }

  document.getElementById("load-prime-btn").addEventListener("click", () => {
    setFormValues(45, 130000, 120000, 800, 0);
  });

  document.getElementById("load-borderline-btn").addEventListener("click", () => {
    setFormValues(36, 62000, 220000, 650, 1);
  });

  document.getElementById("load-subprime-btn").addEventListener("click", () => {
    setFormValues(42, 65000, 350000, 580, 2);
  });

  document.getElementById("reset-form-btn").addEventListener("click", () => {
    setFormValues(35, 75000, 200000, 680, 0, false);
  });

  // -------------------------------------------------------------------------
  // Client-Side Form Validation (Pydantic Rule Mirroring)
  // -------------------------------------------------------------------------
  function validateForm() {
    let isValid = true;

    const age = parseInt(ageInput.value);
    if (isNaN(age) || age < 18 || age > 100) {
      ageError.classList.add("visible");
      ageInput.classList.add("invalid");
      isValid = false;
    } else {
      ageError.classList.remove("visible");
      ageInput.classList.remove("invalid");
    }

    const income = parseFloat(incomeInput.value);
    if (isNaN(income) || income <= 0) {
      incomeError.classList.add("visible");
      incomeInput.classList.add("invalid");
      isValid = false;
    } else {
      incomeError.classList.remove("visible");
      incomeInput.classList.remove("invalid");
    }

    const loan = parseFloat(loanInput.value);
    if (isNaN(loan) || loan <= 0) {
      loanError.classList.add("visible");
      loanInput.classList.add("invalid");
      isValid = false;
    } else {
      loanError.classList.remove("visible");
      loanInput.classList.remove("invalid");
    }

    const score = parseInt(scoreInput.value);
    if (isNaN(score) || score < 300 || score > 850) {
      scoreError.classList.add("visible");
      scoreInput.classList.add("invalid");
      isValid = false;
    } else {
      scoreError.classList.remove("visible");
      scoreInput.classList.remove("invalid");
    }

    const history = parseInt(historyInput.value);
    if (isNaN(history) || history < 0) {
      historyError.classList.add("visible");
      historyInput.classList.add("invalid");
      isValid = false;
    } else {
      historyError.classList.remove("visible");
      historyInput.classList.remove("invalid");
    }

    return isValid;
  }

  // -------------------------------------------------------------------------
  // Render Underwriting Assessment & Dynamic SHAP Bars
  // -------------------------------------------------------------------------
  function renderAssessment(result, applicantData) {
    const prob = result.default_probability;
    const probPct = (prob * 100).toFixed(1);
    const riskLevel = result.risk_level;

    // 1. Animate Gauge
    probPercentage.textContent = `${probPct}%`;
    const offset = GAUGE_CIRCUMFERENCE - (prob * GAUGE_CIRCUMFERENCE);
    gaugeCircle.style.strokeDashoffset = offset;

    // 2. Risk Tier Styling & Dial Colors
    riskTierPill.className = "risk-tier-pill";
    if (riskLevel === "LOW") {
      riskTierPill.classList.add("tier-low");
      riskTierPill.textContent = "LOW RISK";
      gaugeCircle.style.stroke = "var(--risk-low)";
      decisionActionTitle.textContent = "Fast-Track Approval Recommended";
      decisionRationale.textContent = 
        "The applicant demonstrates pristine repayment behavior and robust debt-coverage headroom. Approved for automated disbursement.";
    } else if (riskLevel === "MEDIUM") {
      riskTierPill.classList.add("tier-med");
      riskTierPill.textContent = "MEDIUM RISK";
      gaugeCircle.style.stroke = "var(--risk-med)";
      decisionActionTitle.textContent = "Secondary Underwriting / Manual Review";
      decisionRationale.textContent = 
        "Moderate credit risk detected. Secondary underwriting required; consider requesting supplementary income proof or a co-borrower.";
    } else {
      riskTierPill.classList.add("tier-high");
      riskTierPill.textContent = "HIGH RISK";
      gaugeCircle.style.stroke = "var(--risk-high)";
      decisionActionTitle.textContent = "Adverse Action Notice (Decline)";
      decisionRationale.textContent = 
        "High default probability detected. In compliance with Fair Lending standards, key adverse risk factors are outlined below for the applicant notice.";
    }

    // 3. Update Audit Badge
    if (result.borrower_id && result.prediction_id) {
      auditIdBadge.textContent = `Borrower #${result.borrower_id} | Audit #${result.prediction_id}`;
    }

    // 4. Render Top 3 SHAP Attributions
    shapContainer.innerHTML = "";
    const factors = result.top_risk_factors || [];

    // Find max absolute SHAP for relative bar width
    const maxAbsShap = Math.max(...factors.map(f => Math.abs(f.shap_value)), 1.0);

    factors.forEach((factor, index) => {
      const isAdverse = factor.impact === "adverse";
      const barPct = Math.min(Math.round((Math.abs(factor.shap_value) / maxAbsShap) * 100), 100);

      const card = document.createElement("div");
      card.className = "shap-card";
      card.innerHTML = `
        <div class="shap-card-header">
          <div class="shap-feature-name">
            <span>${index + 1}.</span>
            <span>${formatFeatureName(factor.feature)}</span>
            <span style="font-size: 0.76rem; color: var(--text-muted); font-weight: normal;">(val: ${factor.feature_value})</span>
          </div>
          <span class="shap-impact-tag ${isAdverse ? 'impact-adverse' : 'impact-protective'}">
            ${isAdverse ? '+ ' : '- '}SHAP: ${factor.shap_value.toFixed(3)} (${factor.impact})
          </span>
        </div>
        <div class="shap-bar-bg">
          <div class="shap-bar-fill ${isAdverse ? 'fill-adverse' : 'fill-protective'}" style="width: 0%;"></div>
        </div>
        <div class="shap-narrative">
          ${factor.description}
        </div>
      `;

      shapContainer.appendChild(card);

      // Trigger width animation on next tick
      setTimeout(() => {
        const fill = card.querySelector(".shap-bar-fill");
        if (fill) fill.style.width = `${Math.max(barPct, 8)}%`;
      }, 50 * index + 50);
    });

    // 5. Add to Audit Table
    addAuditTableRow({
      id: result.prediction_id || sessionAuditRecords.length + 1,
      age: applicantData.age,
      income: applicantData.income,
      loan: applicantData.loan_amount,
      score: applicantData.credit_score,
      defaults: applicantData.default_history,
      prob: (prob * 100).toFixed(1) + "%",
      tier: riskLevel,
      primaryFactor: factors[0] ? formatFeatureName(factors[0].feature) : "N/A",
      timestamp: new Date().toLocaleTimeString(),
    });
  }

  function formatFeatureName(name) {
    const map = {
      "Age": "Borrower Age",
      "Income": "Annual Income",
      "LoanAmount": "Loan Principal Size",
      "CreditScore": "Credit Bureau Score",
      "DefaultHistory": "Prior Default Count",
    };
    return map[name] || name;
  }

  // -------------------------------------------------------------------------
  // Add to Audit Log Table
  // -------------------------------------------------------------------------
  function addAuditTableRow(record) {
    sessionAuditRecords.unshift(record);
    const row = document.createElement("tr");

    let tierClass = "tier-high";
    if (record.tier === "LOW") tierClass = "tier-low";
    else if (record.tier === "MEDIUM") tierClass = "tier-med";

    row.innerHTML = `
      <td><strong>#${record.id}</strong></td>
      <td>${record.age} yrs</td>
      <td>${formatCurrency(record.income)}</td>
      <td>${formatCurrency(record.loan)}</td>
      <td><strong>${record.score}</strong></td>
      <td>${record.defaults}</td>
      <td><strong>${record.prob}</strong></td>
      <td><span class="risk-tier-pill ${tierClass}" style="font-size: 0.72rem; padding: 2px 8px;">${record.tier}</span></td>
      <td>${record.primaryFactor}</td>
      <td style="color: var(--text-muted); font-size: 0.74rem;">${record.timestamp}</td>
    `;

    // Row click reloads borrower
    row.addEventListener("click", () => {
      setFormValues(record.age, record.income, record.loan, record.score, record.defaults);
    });

    auditTableBody.insertBefore(row, auditTableBody.firstChild);
  }

  // -------------------------------------------------------------------------
  // Form Submit Handler (Calling POST /predict)
  // -------------------------------------------------------------------------
  async function handleFormSubmit(e) {
    if (e) e.preventDefault();

    if (!validateForm()) {
      return;
    }

    const payload = {
      age: parseInt(ageInput.value),
      income: parseFloat(incomeInput.value),
      loan_amount: parseFloat(loanInput.value),
      credit_score: parseInt(scoreInput.value),
      default_history: parseInt(historyInput.value),
    };

    submitBtn.classList.add("loading");
    document.getElementById("submit-btn-text").textContent = "Calculating SHAP Attributions...";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      renderAssessment(result, payload);

    } catch (err) {
      console.error("Underwriting prediction failed:", err);
      alert(`Underwriting Error: ${err.message}`);
    } finally {
      submitBtn.classList.remove("loading");
      document.getElementById("submit-btn-text").textContent = "Execute Underwriting & SHAP Scoring";
    }
  }

  underwritingForm.addEventListener("submit", handleFormSubmit);

  // -------------------------------------------------------------------------
  // Initial API Health Check & Existing Decisions
  // -------------------------------------------------------------------------
  async function loadInitialState() {
    try {
      const healthRes = await fetch("/health");
      if (healthRes.ok) {
        const health = await healthRes.json();
        const statusText = document.getElementById("system-status-text");
        if (health.model_loaded && health.explainer_loaded) {
          statusText.textContent = `Engine Active (XGBoost & SHAP) • DB: ${health.database_connected ? "Connected" : "Simulated"}`;
        }
      }

      // Fetch model metrics
      const metricsRes = await fetch("/metrics");
      if (metricsRes.ok) {
        const metrics = await metricsRes.json();
        if (metrics.evaluation_metrics) {
          document.getElementById("metric-roc-auc").textContent = metrics.evaluation_metrics.roc_auc.toFixed(4);
          document.getElementById("metric-pr-auc").textContent = metrics.evaluation_metrics.pr_auc.toFixed(4);
          document.getElementById("metric-recall").textContent = (metrics.evaluation_metrics.recall * 100).toFixed(2) + "%";
          document.getElementById("metric-precision").textContent = (metrics.evaluation_metrics.precision * 100).toFixed(2) + "%";
        }
      }

      // Trigger initial baseline analysis with prime applicant
      setFormValues(42, 65000, 350000, 580, 2);

    } catch (err) {
      console.warn("Could not query initial health/metrics:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Modal Interactions
  // -------------------------------------------------------------------------
  openMetricsBtn.addEventListener("click", () => {
    metricsModal.classList.add("open");
  });

  closeModalBtn.addEventListener("click", () => {
    metricsModal.classList.remove("open");
  });

  metricsModal.addEventListener("click", (e) => {
    if (e.target === metricsModal) {
      metricsModal.classList.remove("open");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && metricsModal.classList.contains("open")) {
      metricsModal.classList.remove("open");
    }
  });

  loadInitialState();
});
