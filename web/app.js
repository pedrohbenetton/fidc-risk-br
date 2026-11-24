/* FIDC-RISK.BR — Monochrome Futuristic Telemetry Engine (Zero Dependencies, HiDPI Reticle Canvas) */
(() => {
  "use strict";

  const MODEL_NAMES = {
    heuristic_committee: "Comitê Estrutural [Regra]",
    logistic_hazard_va: "Hazard Logístico + Venn-Abers",
    lgbm_unconstrained_raw: "LightGBM Irrestrito [Bruto]",
    lgbm_unconstrained_platt: "LightGBM + Platt Scaling",
    lgbm_unconstrained_isotonic: "LightGBM + Isotônica [PAV]",
    lgbm_unconstrained_va: "LightGBM Irrestrito + VA",
    lgbm_monotonic_va: "LightGBM Monotônico + VA [API]",
    catboost_ordered_va: "CatBoost Ordenado + VA",
    gbdt_ensemble_va: "Ensemble Tabular [LGBM+CB] + VA",
    temporal_transformer_5seed_va: "Transformer Causal [GPU · 5 Seeds]",
    hetero_contagion_gnn_5seed_va: "HeteroContagionGNN LOO [GPU · 5 Seeds]",
    hybrid_gbdt_contagion_gnn: "Híbrido: GBDT + rede de contágio (GNN)",
  };

  const POLICY_NAMES = {
    benchmark_cdi: "100% do CDI",
    equal_weight_senior: "Peso igual em cotas sênior",
    yield_chasing_top25: "25 maiores retornos, sem filtro",
    traditional_committee: "Regra de comitê (subord. ≥ 15%, vencido ≤ 3%)",
    ml_watchlist_cutoff: "Lista do modelo com corte (t+2)",
    cvar_venn_abers_lagged: "Modelo com controle de cauda (t+2)",
    cvar_venn_abers_frictionless_t1: "Referência sem atraso de resgate (t+1)",
  };

  const fmtInt = (n) => Number(n || 0).toLocaleString("pt-BR");
  const fmtPct = (v, d = 1) => `${Number(v || 0).toFixed(d).replace(".", ",")}%`;
  const fmtNum = (v, d = 2) => Number(v || 0).toFixed(d).replace(".", ",");

  // Build demo (fundos anonimizados): sem API ao vivo e sem pacote nominal; PL só em faixas.
  const IS_DEMO = window.FIDC_DEMO === true;
  const BAND_MID_MI = [5, 17, 35, 70, 160, 350, 700, 1600, 4000];
  const idLabel = (f) => f.cnpj || f.entity_id;
  const plLabel = (f) => (f.pl_faixa ? f.pl_faixa : `R$ ${fmtNum(f.pl_m, 1)} mi`);
  // Faixas de PL na coluna "PL (R$ mi)", na mesma ordem de export/anonymize.py (PL_BANDS_MI).
  const PL_BAND_SHORT = ["até 10", "10–25", "25–50", "50–100", "100–250", "250–500", "500–1.000", "1.000–2.500", "> 2.500"];
  const plCell = (f) => (f.pl_faixa ? PL_BAND_SHORT[f.pl_ordem] || f.pl_faixa : fmtNum(f.pl_m, 1));
  const plSortValue = (f) => (f.pl_m != null ? f.pl_m : BAND_MID_MI[f.pl_ordem || 0]);
  const exposureLabel = (n) => (n.exposicao_faixa ? n.exposicao_faixa : `R$ ${fmtNum(n.exposure_m, 1)} mi`);

  let BUNDLE = null;
  let SELECTED_FUND = null;
  let API_AVAILABLE = false;
  let TRAJ_HOVER_IDX = null;
  let WEALTH_HOVER_IDX = null;

  const GRAPH_STATE = {
    nodes: [],
    links: [],
    selectedId: null,
    hoveredId: null,
    minPdPct: 0,
    zoom: 1.0,
  };

  function setupHiDPICanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width > 20 ? rect.width : Number(canvas.getAttribute("width")) || 560;
    const cssH = Number(canvas.getAttribute("height")) || 220;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.height = `${cssH}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: cssW, h: cssH };
  }

  function createDiagonalHatchPattern(ctx, strokeColor = "rgba(27,36,51,0.28)") {
    const pCanvas = document.createElement("canvas");
    pCanvas.width = 8;
    pCanvas.height = 8;
    const pCtx = pCanvas.getContext("2d");
    pCtx.strokeStyle = strokeColor;
    pCtx.lineWidth = 1;
    pCtx.beginPath();
    pCtx.moveTo(0, 8);
    pCtx.lineTo(8, 0);
    pCtx.stroke();
    return ctx.createPattern(pCanvas, "repeat");
  }

  function agingRuler(f) {
    // Régua da carteira: em dia | atraso 1–60 (âmbar) | 61–90 dias | vencido 90+ (vermelho).
    const late = Math.max(0, Math.min(100, Number(f.inad90_pct) || 0));
    const mid = Math.max(0, Math.min(100 - late, Number(f.late_61_90_pct) || 0));
    const early = Math.max(0, Math.min(100 - late - mid, Number(f.early_pct) || 0));
    const ok = Math.max(0, 100 - late - mid - early);
    const title = `Vencido 90+ dias: ${fmtPct(late, 1)} · 61–90 dias: ${fmtPct(mid, 1)} · atraso 1–60 dias: ${fmtPct(early, 1)}`;
    return `<span class="aging-ruler" role="img" aria-label="${title}" title="${title}">` +
      (ok > 0 ? `<i class="seg-ok" style="flex:${ok}"></i>` : "") +
      (early > 0 ? `<i class="seg-early" style="flex:${early}"></i>` : "") +
      (mid > 0 ? `<i class="seg-6190" style="flex:${mid}"></i>` : "") +
      (late > 0 ? `<i class="seg-late" style="flex:${late}"></i>` : "") +
      `</span>`;
  }

  function riskClass(pd) {
    if (pd >= 0.20) return "risk-high";
    if (pd >= 0.08) return "risk-med";
    return "risk-low";
  }

  async function checkLiveApi() {
    try {
      const resp = await fetch("/health", { method: "GET" });
      if (resp.ok) {
        const js = await resp.json();
        API_AVAILABLE = Boolean(js && js.model_loaded);
      }
    } catch (_) {
      API_AVAILABLE = false;
    }
  }

  function activateTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.toggle("active", p.id === `tab-${tabId}`);
    });
    if (tabId === "radar" && SELECTED_FUND) drawFundTrajectoryCanvas(SELECTED_FUND);
    if (tabId === "contagion") drawContagionGraph();
    if (tabId === "simulator") renderSimulator();
    if (tabId === "science") {
      drawAblationCanvas();
      drawEventStudyAndReliability();
    }
  }

  function populateKPIs() {
    const k = BUNDLE.kpis;
    const m = BUNDLE.meta;
    document.getElementById("mode-indicator").textContent = API_AVAILABLE
      ? "API ao vivo"
      : "Dados públicos da CVM";
    document.getElementById("header-meta").textContent =
      `Competências de ${m.min_competence} a ${m.latest_competence} · ${m.months_processed} meses · ${k.feature_count} variáveis`;

    document.getElementById("kpi-rows").textContent = fmtInt(k.fund_month_rows);
    document.getElementById("kpi-entities").textContent =
      `${fmtInt(k.unique_entities)} FIDCs · gestor identificado em ${fmtPct(k.gestor_coverage_rate * 100, 1)}`;

    document.getElementById("kpi-edges").textContent = fmtInt(k.cedente_edge_rows);
    document.getElementById("kpi-cedentes").textContent =
      `${fmtInt(k.unique_pj_cedentes)} cedentes PJ distintos`;

    document.getElementById("kpi-pl").textContent = `R$ ${fmtNum(k.live_total_pl_bi, 1)} bi`;
    document.getElementById("kpi-live-funds").textContent =
      `${fmtInt(k.live_commercial_funds)} fundos ativos em ${m.latest_competence}`;

    // Métricas do modelo que gera os scores mostrados (o servido), não do melhor modelo do benchmark.
    const rows = BUNDLE.benchmarks_h6 || [];
    const served = rows.find((r) => r.model === "lgbm_monotonic_va") || [...rows].sort((a, b) => b.pr_auc - a.pr_auc)[0];
    const comite = rows.find((r) => r.model === "heuristic_committee");
    if (served) {
      document.getElementById("kpi-prauc").textContent = fmtNum(served.pr_auc, 4);
      document.getElementById("kpi-ece").textContent =
        `Regra de comitê: ${comite ? fmtNum(comite.pr_auc, 2) : "—"} · ROC ${fmtNum(served.roc_auc, 3)} · lift ${fmtNum(served.lift_top5, 2)}×`;
    }

    const vaPol = (BUNDLE.portfolio_summary || []).find((p) => p.policy === "cvar_venn_abers_lagged");
    if (vaPol) {
      document.getElementById("kpi-sharpe").textContent = `${fmtNum(vaPol.pct_of_cdi, 1)}% do CDI`;
      document.getElementById("kpi-cdi-pct").textContent =
        `Backtest 2022–2026 · ${vaPol.severe_monthly_shocks_suffered} choques severos`;
    }
  }

  /* ================= COMMAND PALETTE (Ctrl+K or /) ================= */
  function initCommandPalette() {
    const backdrop = document.getElementById("cmd-backdrop");
    const input = document.getElementById("cmd-input");
    const results = document.getElementById("cmd-results");

    const openCmd = () => {
      backdrop.classList.remove("hidden");
      input.value = "";
      renderCmdList("");
      setTimeout(() => input.focus(), 20);
    };
    const closeCmd = () => backdrop.classList.add("hidden");

    document.getElementById("btn-open-cmd").addEventListener("click", openCmd);
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeCmd();
    });

    document.addEventListener("keydown", (e) => {
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if ((e.key === "/" && tag !== "INPUT" && tag !== "SELECT") || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k")) {
        e.preventDefault();
        openCmd();
      } else if (e.key === "Escape") {
        closeCmd();
      }
    });

    input.addEventListener("input", () => renderCmdList(input.value.trim().toLowerCase()));

    function renderCmdList(q) {
      const matches = BUNDLE.watchlist_live
        .filter((f) => !q || `${f.name} ${idLabel(f)} ${f.gestor} ${f.sector}`.toLowerCase().includes(q))
        .slice(0, 12);

      results.innerHTML = matches
        .map(
          (f) => `
          <div class="cmd-item" data-eid="${f.entity_id}">
            <div>
              <div style="color:#1b2433;font-weight:500;font-size:12px">${f.name.slice(0, 56)}</div>
              <div style="font-family:var(--font-mono);font-size:10px;color:#7a7f88">${idLabel(f)} · ${f.sector} · PL ${plLabel(f)}</div>
            </div>
            <span class="risk-pill ${riskClass(f.prob_6m)}">${fmtPct(f.prob_6m * 100, 1)}</span>
          </div>`
        )
        .join("");

      results.querySelectorAll(".cmd-item").forEach((el) => {
        el.addEventListener("click", () => {
          const eid = el.getAttribute("data-eid");
          const found = BUNDLE.watchlist_live.find((x) => x.entity_id === eid);
          closeCmd();
          if (found) {
            activateTab("radar");
            selectFund(found);
          }
        });
      });
    }
  }

  /* ================= TAB 01: WATCHLIST & DOSSIER ================= */
  function initWatchlistControls() {
    const sectors = [...new Set(BUNDLE.watchlist_live.map((f) => f.sector))].sort();
    const secSel = document.getElementById("sector-filter");
    sectors.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      secSel.appendChild(opt);
    });

    ["search-input", "sector-filter", "condom-filter", "sort-select"].forEach((id) => {
      const el = document.getElementById(id);
      el.addEventListener("input", renderWatchlistTable);
      el.addEventListener("change", renderWatchlistTable);
    });

    ["sl-sh-early", "sl-sh-subord", "sl-sh-cont"].forEach((id) => {
      document.getElementById(id).addEventListener("input", updateStressSimulation);
    });
    document.getElementById("stress-reset-btn").addEventListener("click", () => {
      document.getElementById("sl-sh-early").value = "0";
      document.getElementById("sl-sh-subord").value = "0";
      document.getElementById("sl-sh-cont").value = "0";
      updateStressSimulation();
    });

    // Interactive Hover Reticle on Trajectory Canvas
    const tCanvas = document.getElementById("trajectory-canvas");
    const tTip = document.getElementById("traj-tooltip");
    tCanvas.addEventListener("mousemove", (ev) => {
      if (!SELECTED_FUND || !SELECTED_FUND.history || !SELECTED_FUND.history.length) return;
      const rect = tCanvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const padL = 40;
      const padR = 22;
      const totalSteps = SELECTED_FUND.history.length + 2;
      const frac = (mx - padL) / Math.max(rect.width - padL - padR, 1);
      const idx = Math.round(frac * (totalSteps - 1));
      if (idx >= 0 && idx < SELECTED_FUND.history.length) {
        TRAJ_HOVER_IDX = idx;
        const pt = SELECTED_FUND.history[idx];
        tTip.classList.remove("hidden");
        tTip.style.left = `${Math.min(rect.width - 175, Math.max(8, mx + 10))}px`;
        tTip.style.top = "10px";
        tTip.innerHTML = `${pt.dt} // &gt;90D: ${fmtPct(pt.inad90, 2)} · 1-60D: ${fmtPct(pt.early, 2)} · SUB: ${fmtPct(pt.subord, 1)}`;
        drawFundTrajectoryCanvas(SELECTED_FUND);
      } else if (idx >= SELECTED_FUND.history.length) {
        TRAJ_HOVER_IDX = totalSteps - 1;
        tTip.classList.remove("hidden");
        tTip.style.left = `${Math.max(8, rect.width - 220)}px`;
        tTip.style.top = "10px";
        tTip.innerHTML = `Daqui a 3 meses · provável ${fmtPct(SELECTED_FUND.cqr_h3_q50, 2)} · faixa ${fmtPct(SELECTED_FUND.cqr_h3_q10, 2)} a ${fmtPct(SELECTED_FUND.cqr_h3_q90, 2)}`;
        drawFundTrajectoryCanvas(SELECTED_FUND);
      }
    });
    tCanvas.addEventListener("mouseleave", () => {
      TRAJ_HOVER_IDX = null;
      tTip.classList.add("hidden");
      if (SELECTED_FUND) drawFundTrajectoryCanvas(SELECTED_FUND);
    });
  }

  function renderWatchlistTable() {
    const q = document.getElementById("search-input").value.trim().toLowerCase();
    const sec = document.getElementById("sector-filter").value;
    const cond = document.getElementById("condom-filter").value;
    const sortKey = document.getElementById("sort-select").value;

    const rows = BUNDLE.watchlist_live.filter((f) => {
      if (sec && f.sector !== sec) return false;
      if (cond === "open" && f.is_aberto !== 1) return false;
      if (cond === "closed" && f.is_aberto !== 0) return false;
      if (q) {
        const hay = `${f.name} ${idLabel(f)} ${f.entity_id} ${f.gestor}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    rows.sort((a, b) => (sortKey === "pl_m" ? plSortValue(b) - plSortValue(a) : (b[sortKey] || 0) - (a[sortKey] || 0)));
    document.getElementById("watchlist-count").textContent = `${fmtInt(rows.length)} fundos`;

    const tbody = document.getElementById("watchlist-tbody");
    tbody.innerHTML = "";

    rows.slice(0, 180).forEach((f) => {
      const tr = document.createElement("tr");
      if (SELECTED_FUND && SELECTED_FUND.entity_id === f.entity_id) {
        tr.classList.add("selected");
      }
      tr.innerHTML = `
        <td>
          <span class="fund-cell-title" title="${f.name}">${f.name}</span>
          <span class="fund-cell-sub">${f.gestor.slice(0, 28)} · ${idLabel(f)}</span>
        </td>
        <td class="cell-sector">${f.sector}</td>
        <td class="num">${plCell(f)}</td>
        <td class="num">
          <span class="risk-pill ${riskClass(f.prob_6m)}">${fmtPct(f.prob_6m * 100, 1)}</span>
          <div class="fund-cell-sub">[${fmtPct(f.va_p0 * 100, 0)}–${fmtPct(f.va_p1 * 100, 0)}]</div>
        </td>
        <td class="num"><div class="aging-cell">${agingRuler(f)}<span>${fmtPct(f.inad90_pct, 1)}</span></div></td>
        <td class="num">${fmtPct(f.subord_pct, 1)}</td>
        <td class="num">${fmtNum(f.recompra_idx, 2)}</td>
      `;
      tr.addEventListener("click", () => selectFund(f));
      tbody.appendChild(tr);
    });

    if (!SELECTED_FUND && rows.length > 0) {
      selectFund(rows[0]);
    }
  }

  function renderLedMeter(pd) {
    const el = document.getElementById("insp-led-meter");
    const activeCount = Math.min(10, Math.max(1, Math.round(pd * 14)));
    let html = "";
    for (let i = 0; i < 10; i++) {
      html += `<span class="led-seg ${i < activeCount ? "on" : ""}"></span>`;
    }
    el.innerHTML = html;
  }

  function selectFund(fund) {
    SELECTED_FUND = fund;
    TRAJ_HOVER_IDX = null;
    document.querySelectorAll("#watchlist-tbody tr").forEach((tr) => {
      const sub = tr.querySelector(".fund-cell-sub");
      tr.classList.toggle("selected", Boolean(sub && sub.textContent.includes(idLabel(fund))));
    });

    document.getElementById("insp-sector").textContent = fund.sector;
    document.getElementById("insp-condom").textContent = fund.is_aberto ? "Aberto" : "Fechado";
    document.getElementById("insp-name").textContent = fund.name;
    document.getElementById("insp-meta").textContent =
      `${fund.gestor} · PL ${plLabel(fund)} · ${idLabel(fund)}`;

    document.getElementById("insp-pd").textContent = fmtPct(fund.prob_6m * 100, 1);
    renderLedMeter(fund.prob_6m);
    document.getElementById("insp-va-band").textContent =
      `Faixa ${fmtPct(fund.va_p0 * 100, 0)}–${fmtPct(fund.va_p1 * 100, 0)}`;

    document.getElementById("insp-inad90").textContent = fmtPct(fund.inad90_pct, 2);
    document.getElementById("insp-early").textContent = fmtPct(fund.early_pct, 2);
    document.getElementById("insp-subord").textContent = fmtPct(fund.subord_pct, 1);
    document.getElementById("insp-pdd").textContent = fmtPct(fund.pdd_pct, 2);
    document.getElementById("insp-recompra").textContent = fmtNum(fund.recompra_idx, 2);
    document.getElementById("insp-contagion").textContent = fmtPct(fund.contagion_pct, 2);

    drawFundTrajectoryCanvas(fund);
    renderShapBipolar(fund.shap || []);
    updateStressSimulation();
  }

  function drawFundTrajectoryCanvas(fund) {
    const canvas = document.getElementById("trajectory-canvas");
    const { ctx, w, h } = setupHiDPICanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const hist = fund.history || [];
    if (!hist.length) return;

    const pad = { l: 40, r: 30, t: 16, b: 28 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    const allVals = [
      ...hist.map((p) => p.inad90),
      ...hist.map((p) => p.early),
      ...hist.map((p) => Math.min(p.subord, 65)),
      fund.cqr_h3_q90,
      8,
    ];
    const yMax = Math.max(...allVals) * 1.14;
    const totalSteps = hist.length + 2;
    const xAt = (i) => pad.l + (i / Math.max(totalSteps - 1, 1)) * pw;
    const yAt = (v) => pad.t + ph - (Math.min(Math.max(v, 0), yMax) / yMax) * ph;

    // Monochrome dotted precision grid
    ctx.strokeStyle = "rgba(27,36,51,0.08)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#7a7f88";
    ctx.font = "11px 'IBM Plex Sans', monospace";
    for (let g = 0; g <= 4; g++) {
      const val = (yMax * g) / 4;
      const yy = yAt(val);
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(w - pad.r, yy);
      ctx.stroke();
      ctx.fillText(`${val.toFixed(0)}%`, 4, yy + 3);
    }

    const lastIdx = hist.length - 1;
    const xNow = xAt(lastIdx);
    const xFut = xAt(totalSteps - 1);

    // Origin divider line
    ctx.save();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = "rgba(31,58,95,0.35)";
    ctx.beginPath();
    ctx.moveTo(xNow, pad.t);
    ctx.lineTo(xNow, h - pad.b);
    ctx.stroke();
    ctx.restore();

    // CQR Conformal Cone [q10, q90] in architectural diagonal hatch
    const yNowInad = yAt(hist[lastIdx].inad90);
    ctx.save();
    ctx.fillStyle = createDiagonalHatchPattern(ctx, "rgba(31,58,95,0.30)");
    ctx.beginPath();
    ctx.moveTo(xNow, yNowInad);
    ctx.lineTo(xFut, yAt(fund.cqr_h3_q90));
    ctx.lineTo(xFut, yAt(fund.cqr_h3_q10));
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "rgba(27,36,51,0.45)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // CQR Median Projection
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#8e2323";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(xNow, yNowInad);
    ctx.lineTo(xFut, yAt(fund.cqr_h3_q50));
    ctx.stroke();
    ctx.restore();

    // Subordinação (dark etched line)
    ctx.strokeStyle = "#7a7f88";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    hist.forEach((pt, i) => {
      const xx = xAt(i);
      const yy = yAt(Math.min(pt.subord, 65));
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();

    // Atraso 1-60d (silver dashed)
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "#d9902b";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    hist.forEach((pt, i) => {
      const xx = xAt(i);
      const yy = yAt(pt.early);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
    ctx.restore();

    // Inadimplência >90d (Pure stark white + subtle luminescence)
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0)";
    ctx.shadowBlur = 7;
    ctx.strokeStyle = "#8e2323";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    hist.forEach((pt, i) => {
      const xx = xAt(i);
      const yy = yAt(pt.inad90);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
    ctx.restore();

    // Hover Reticle Crosshair
    if (TRAJ_HOVER_IDX !== null) {
      const hx = xAt(TRAJ_HOVER_IDX);
      ctx.strokeStyle = "rgba(31,58,95,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, pad.t);
      ctx.lineTo(hx, h - pad.b);
      ctx.stroke();
    }

    ctx.save();
    ctx.fillStyle = "#7a7f88";
    ctx.font = "11px 'IBM Plex Sans', monospace";
    ctx.textAlign = "left";
    ctx.fillText(hist[0].dt, pad.l, h - 8);
    ctx.textAlign = "right";
    ctx.fillText(hist[lastIdx].dt, xNow - 6, h - 8);
    ctx.fillStyle = "#1b2433";
    ctx.textAlign = "left";
    ctx.fillText("+3 meses", xNow + 6, h - 8);
    ctx.restore();
  }

  function renderShapBipolar(drivers) {
    const wrap = document.getElementById("shap-bars");
    wrap.innerHTML = "";
    if (!drivers.length) {
      wrap.innerHTML = "<span class='mono-hint'>Nenhum fator relevante para este fundo.</span>";
      return;
    }
    const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap_log_odds)), 0.05);
    drivers.slice(0, 6).forEach((d) => {
      const val = d.shap_log_odds;
      const halfPct = Math.min(49, Math.round((Math.abs(val) / maxAbs) * 49));
      const posStyle = val >= 0 ? `left:50%;width:${halfPct}%` : `left:${50 - halfPct}%;width:${halfPct}%`;
      const row = document.createElement("div");
      row.className = "shap-bi-row";
      row.innerHTML = `
        <span class="shap-bi-lbl" title="${d.label}${d.feature_value != null ? ` (valor=${d.feature_value})` : ""}">${d.label}</span>
        <div class="shap-bi-axis">
          <div class="shap-bi-bar ${val >= 0 ? "shap-bi-pos" : "shap-bi-neg"}" style="${posStyle}"></div>
        </div>
        <span class="shap-bi-val" style="color:${val >= 0 ? "#1b2433" : "#4a5160"}">${val >= 0 ? "+" : ""}${val.toFixed(3)}</span>
      `;
      wrap.appendChild(row);
    });
  }

  async function updateStressSimulation() {
    if (!SELECTED_FUND) return;
    const dEarly = parseFloat(document.getElementById("sl-sh-early").value);
    const dSub = parseFloat(document.getElementById("sl-sh-subord").value);
    const dCont = parseFloat(document.getElementById("sl-sh-cont").value);

    document.getElementById("lbl-sh-early").textContent = `+${fmtNum(dEarly, 1)} pp`;
    document.getElementById("lbl-sh-subord").textContent = `${dSub >= 0 ? "+" : ""}${fmtNum(dSub, 1)} pp`;
    document.getElementById("lbl-sh-cont").textContent = `+${fmtNum(dCont, 1)} pp`;

    const resBox = document.getElementById("stress-result");
    if (dEarly === 0 && dSub === 0 && dCont === 0) {
      resBox.innerHTML = `Hoje: <b>${fmtPct(SELECTED_FUND.prob_6m * 100, 1)}</b> de risco em 6 meses · atraso 90d+ projetado para 3 meses: <b>${fmtPct(SELECTED_FUND.cqr_h3_q50, 2)}</b>`;
      return;
    }

    const newEarly = Math.min(1.0, Math.max(0.0, SELECTED_FUND.early_pct / 100.0 + dEarly / 100.0));
    const newInadTotal = Math.min(1.0, Math.max(0.0, (SELECTED_FUND.inad90_pct + SELECTED_FUND.early_pct + dEarly) / 100.0));
    const newSub = Math.min(1.0, Math.max(0.0, SELECTED_FUND.subord_pct / 100.0 + dSub / 100.0));
    const newNetCushion = Math.max(-1.0, Math.min(1.0, newSub - Math.max(0.0, SELECTED_FUND.inad90_pct / 100.0 - SELECTED_FUND.pdd_pct / 100.0)));
    const newCont = Math.min(1.0, Math.max(0.0, SELECTED_FUND.contagion_pct / 100.0 + dCont / 100.0));

    if (API_AVAILABLE) {
      try {
        const resp = await fetch("/api/v1/score", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            entity_id: SELECTED_FUND.entity_id,
            feature_overrides: {
              inad_early_1_60_pct: newEarly,
              inad_early_1_90_pct: newEarly,
              inad_31_60_pct: newEarly * 0.6,
              inad_total_pct: newInadTotal,
              subordinacao_pct: newSub,
              total_enhancement_pct: newSub,
              net_cushion_over_bad_debt: newNetCushion,
              contagion_cedente_inad90p_wt: newCont,
              contagion_cedente_stress_wt: Math.min(1.0, newCont * 1.5),
            },
          }),
        });
        if (resp.ok) {
          const js = await resp.json();
          const sign = js.delta_prob_6m >= 0 ? "+" : "";
          resBox.innerHTML =
            `Risco em 6 meses: <b>${fmtPct(js.base_prob_6m * 100, 1)}</b> → ` +
            `<b class="shock-val">${fmtPct(js.shocked_prob_6m * 100, 1)}</b> ` +
            `(${sign}${fmtPct(js.delta_prob_6m * 100, 1)} pp) · faixa ${fmtPct(js.shocked_va_p0 * 100, 0)}–${fmtPct(js.shocked_va_p1 * 100, 0)}`;
          return;
        }
      } catch (_) {
        /* Fallback estático */
      }
    }

    const p0 = Math.min(Math.max(SELECTED_FUND.prob_6m, 1e-4), 0.999);
    const baseLogit = Math.log(p0 / (1 - p0));
    const deltaLogit = 0.085 * dEarly - 0.055 * dSub + 0.062 * dCont;
    const pShock = 1 / (1 + Math.exp(-(baseLogit + deltaLogit)));
    const diffPp = (pShock - p0) * 100;
    resBox.innerHTML =
      `Risco em 6 meses: <b>${fmtPct(p0 * 100, 1)}</b> → ` +
      `<b class="shock-val">${fmtPct(pShock * 100, 1)}</b> (${diffPp >= 0 ? "+" : ""}${fmtNum(diffPp, 1)} pp)`;
  }

  /* ================= TAB 02: PALANTIR-STYLE CONTAGION TOPOLOGY ================= */
  function initContagionGraph() {
    const raw = BUNDLE.contagion_graph || { nodes: [], links: [] };
    const canvas = document.getElementById("contagion-canvas");
    const tip = document.getElementById("graph-tooltip");
    const w = 900;
    const h = 560;

    const nodes = raw.nodes.map((n, i) => {
      const angle = (i / Math.max(raw.nodes.length, 1)) * Math.PI * 2;
      const radius = n.kind === "cedente" ? 115 + (i % 5) * 30 : 215 + (i % 7) * 22;
      return {
        ...n,
        x: w / 2 + Math.cos(angle) * radius,
        y: h / 2 + Math.sin(angle) * radius * 0.78,
        vx: 0,
        vy: 0,
      };
    });
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    const links = raw.links
      .filter((l) => nodeMap.has(l.source) && nodeMap.has(l.target))
      .map((l) => ({ ...l, sNode: nodeMap.get(l.source), tNode: nodeMap.get(l.target) }));

    for (let iter = 0; iter < 150; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist2 = Math.max(dx * dx + dy * dy, 64);
          const rep = 980 / dist2;
          const d = Math.sqrt(dist2);
          const fx = (dx / d) * rep;
          const fy = (dy / d) * rep;
          nodes[i].vx -= fx;
          nodes[i].vy -= fy;
          nodes[j].vx += fx;
          nodes[j].vy += fy;
        }
      }
      links.forEach((l) => {
        const dx = l.tNode.x - l.sNode.x;
        const dy = l.tNode.y - l.sNode.y;
        const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const force = (d - 52) * 0.045;
        const fx = (dx / d) * force;
        const fy = (dy / d) * force;
        l.sNode.vx += fx;
        l.sNode.vy += fy;
        l.tNode.vx -= fx;
        l.tNode.vy -= fy;
      });
      nodes.forEach((n) => {
        n.vx += (w / 2 - n.x) * 0.012;
        n.vy += (h / 2 - n.y) * 0.015;
        n.x = Math.min(w - 34, Math.max(34, n.x + n.vx * 0.35));
        n.y = Math.min(h - 34, Math.max(34, n.y + n.vy * 0.35));
        n.vx *= 0.4;
        n.vy *= 0.4;
      });
    }

    GRAPH_STATE.nodes = nodes;
    GRAPH_STATE.links = links;

    const slider = document.getElementById("sl-graph-minpd");
    slider.addEventListener("input", () => {
      GRAPH_STATE.minPdPct = parseInt(slider.value, 10);
      document.getElementById("lbl-graph-minpd").textContent = `${GRAPH_STATE.minPdPct}%`;
      drawContagionGraph();
    });
    document.getElementById("btn-graph-recenter").addEventListener("click", () => {
      GRAPH_STATE.selectedId = null;
      GRAPH_STATE.minPdPct = 0;
      GRAPH_STATE.zoom = 1.0;
      slider.value = "0";
      document.getElementById("lbl-graph-minpd").textContent = "0%";
      drawContagionGraph();
    });

    canvas.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const factor = ev.deltaY < 0 ? 1.1 : 0.9;
      GRAPH_STATE.zoom = Math.min(2.4, Math.max(0.65, GRAPH_STATE.zoom * factor));
      drawContagionGraph();
    }, { passive: false });

    const hitTestNode = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = rect.width / 900;
      const scaleY = rect.height / 560;
      const mx = (ev.clientX - rect.left) / scaleX;
      const my = (ev.clientY - rect.top) / scaleY;
      const zx = (mx - 450) / GRAPH_STATE.zoom + 450;
      const zy = (my - 280) / GRAPH_STATE.zoom + 280;

      let hit = null;
      let bestD = 16 / GRAPH_STATE.zoom;
      GRAPH_STATE.nodes.forEach((n) => {
        if (n.risk * 100 < GRAPH_STATE.minPdPct) return;
        const d = Math.hypot(n.x - zx, n.y - zy);
        if (d < bestD) {
          bestD = d;
          hit = n;
        }
      });
      return { hit, px: ev.clientX - rect.left, py: ev.clientY - rect.top, cw: rect.width };
    };

    canvas.addEventListener("mousemove", (ev) => {
      const { hit, px, py, cw } = hitTestNode(ev);
      GRAPH_STATE.hoveredId = hit ? hit.id : null;
      if (hit) {
        tip.classList.remove("hidden");
        tip.style.left = `${Math.min(cw - 240, Math.max(8, px + 14))}px`;
        tip.style.top = `${Math.max(8, py - 28)}px`;
        tip.innerHTML =
          hit.kind === "cedente"
            ? `${hit.label} · ${hit.degree} fundos · exposição ${exposureLabel(hit)}`
            : `${hit.label} // PD ${fmtPct(hit.risk * 100, 1)} · PL ${plLabel(hit)}`;
      } else {
        tip.classList.add("hidden");
      }
      drawContagionGraph();
    });

    canvas.addEventListener("mouseleave", () => {
      GRAPH_STATE.hoveredId = null;
      tip.classList.add("hidden");
      drawContagionGraph();
    });

    canvas.addEventListener("click", (ev) => {
      const { hit } = hitTestNode(ev);
      GRAPH_STATE.selectedId = hit ? hit.id : null;
      renderSelectedGraphNode(hit);
      drawContagionGraph();
    });

    const topCed = nodes.filter((n) => n.kind === "cedente").sort((a, b) => b.risk - a.risk)[0];
    if (topCed) {
      GRAPH_STATE.selectedId = topCed.id;
      renderSelectedGraphNode(topCed);
    }
  }

  function drawContagionGraph() {
    const canvas = document.getElementById("contagion-canvas");
    const { ctx, w, h } = setupHiDPICanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.scale(w / 900, h / 560);

    // Background Orbital Radar Rings & Crosshairs
    ctx.strokeStyle = "rgba(27,36,51,0.035)";
    ctx.lineWidth = 1;
    [95, 185, 275].forEach((rad) => {
      ctx.beginPath();
      ctx.arc(450, 280, rad, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.beginPath();
    ctx.moveTo(450, 0);
    ctx.lineTo(450, 560);
    ctx.moveTo(0, 280);
    ctx.lineTo(900, 280);
    ctx.stroke();

    // Apply Zoom around center (450, 280)
    ctx.translate(450, 280);
    ctx.scale(GRAPH_STATE.zoom, GRAPH_STATE.zoom);
    ctx.translate(-450, -280);

    const focusId = GRAPH_STATE.selectedId || GRAPH_STATE.hoveredId;
    const neighborSet = new Set();
    if (focusId) {
      neighborSet.add(focusId);
      GRAPH_STATE.links.forEach((l) => {
        if (l.source === focusId) neighborSet.add(l.target);
        if (l.target === focusId) neighborSet.add(l.source);
      });
    }

    const vis = (n) => n.risk * 100 >= GRAPH_STATE.minPdPct;

    GRAPH_STATE.links.forEach((l) => {
      if (!vis(l.sNode) || !vis(l.tNode)) return;
      const active = !focusId || (neighborSet.has(l.source) && neighborSet.has(l.target));
      ctx.strokeStyle = active ? "rgba(27,36,51,0.42)" : "rgba(27,36,51,0.06)";
      ctx.lineWidth = active ? Math.min(2.8, 0.7 + l.share_pct / 28) : 0.6;
      ctx.beginPath();
      ctx.moveTo(l.sNode.x, l.sNode.y);
      ctx.lineTo(l.tNode.x, l.tNode.y);
      ctx.stroke();
    });

    GRAPH_STATE.nodes.forEach((n) => {
      if (!vis(n)) return;
      const dimmed = focusId && !neighborSet.has(n.id);
      ctx.globalAlpha = dimmed ? 0.16 : 1.0;

      if (n.kind === "cedente") {
        const sz = Math.min(11, 5.5 + n.degree * 0.85);
        ctx.save();
        ctx.translate(n.x, n.y);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = "#fbfaf7";
        ctx.fillRect(-sz / 2, -sz / 2, sz, sz);
        ctx.strokeStyle = "#1f3a5f";
        ctx.lineWidth = n.id === GRAPH_STATE.selectedId ? 2.2 : 1.3;
        ctx.strokeRect(-sz / 2, -sz / 2, sz, sz);
        ctx.restore();
      } else {
        const r = Math.min(10.5, Math.max(4.0, Math.log10(Math.max(plSortValue(n), 1)) * 2.9));
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        if (n.risk >= 0.20) {
          ctx.fillStyle = "#8e2323";
          ctx.fill();
          // Outer target ring for high-PD nodes
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 3.2, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(142,35,35,0.35)";
          ctx.lineWidth = 1;
          ctx.stroke();
        } else if (n.risk >= 0.08) {
          ctx.fillStyle = "#d9902b";
          ctx.fill();
        } else {
          ctx.fillStyle = "#f1efe8";
          ctx.fill();
          ctx.strokeStyle = "#a9acb2";
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        if (n.id === GRAPH_STATE.selectedId) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 5.5, 0, Math.PI * 2);
          ctx.strokeStyle = "#1f3a5f";
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
      }
    });

    ctx.globalAlpha = 1.0;
    ctx.restore();
  }

  function renderSelectedGraphNode(node) {
    const body = document.getElementById("node-details-body");
    const sub = document.getElementById("node-subtitle");
    if (!node) {
      sub.textContent = "Selecione um vértice na malha topológica para inspecionar exposições cruzadas.";
      body.innerHTML = "";
      return;
    }

    if (node.kind === "cedente") {
      sub.textContent = `${node.label} cede para ${node.degree} fundos · exposição total ${exposureLabel(node)}`;
      const connected = GRAPH_STATE.links
        .filter((l) => l.source === node.id)
        .sort((a, b) => b.tNode.risk - a.tNode.risk);

      body.innerHTML = connected
        .map(
          (l) => `
          <div class="peer-item">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
              <strong style="color:#1b2433;font-size:11.8px">${l.tNode.full_name.slice(0, 44)}</strong>
              <span class="risk-pill ${riskClass(l.tNode.risk)}">${fmtPct(l.tNode.risk * 100, 1)}</span>
            </div>
            <div class="mono-meta" style="margin:5px 0">
              ${idLabel(l.tNode)} · <b>${fmtPct(l.share_pct, 1)}</b> da carteira neste cedente${l.exp_m != null ? ` (R$ ${fmtNum(l.exp_m, 1)} mi)` : ""}<br>
              Vencido 90d+: ${fmtPct(l.tNode.inad90, 1)} · subordinação ${fmtPct(l.tNode.subord, 1)} · ${l.tNode.sector}
            </div>
            <button class="btn-ghost jump-radar-btn" data-eid="${l.tNode.entity_id}" type="button">Abrir fundo</button>
          </div>`
        )
        .join("");
    } else {
      sub.textContent = `${node.full_name} (${idLabel(node)}) // PL ${plLabel(node)} // PD 6M ${fmtPct(node.risk * 100, 1)}`;
      const ceds = GRAPH_STATE.links.filter((l) => l.target === node.id);
      body.innerHTML =
        `<button class="btn-ghost jump-radar-btn" data-eid="${node.entity_id}" type="button" style="align-self:flex-start;margin-bottom:4px">Abrir fundo na carteira</button>` +
        ceds
          .map((l) => {
            const siblings = GRAPH_STATE.links.filter((x) => x.source === l.source && x.target !== node.id);
            return `
            <div class="peer-item">
              <strong style="color:#1b2433">${l.sNode.label}</strong> · <b>${fmtPct(l.share_pct, 1)}</b> da carteira
              <div class="mono-meta" style="margin-top:4px">
                Também cede para outros <b>${siblings.length}</b> fundos (maior risco entre eles: ${fmtPct(l.sNode.risk * 100, 1)}).
              </div>
            </div>`;
          })
          .join("");
    }

    body.querySelectorAll(".jump-radar-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const eid = btn.getAttribute("data-eid");
        const found = BUNDLE.watchlist_live.find((f) => f.entity_id === eid);
        if (found) {
          activateTab("radar");
          selectFund(found);
        }
      });
    });
  }

  /* ================= TAB 03: PORTFOLIO & DECISION SIMULATOR ================= */
  function initSimulator() {
    const slTau = document.getElementById("sl-sim-tau");
    const slCost = document.getElementById("sl-sim-cost");
    const chkFric = document.getElementById("chk-show-frictionless");

    [slTau, slCost, chkFric].forEach((el) => {
      el.addEventListener("input", renderSimulator);
      el.addEventListener("change", renderSimulator);
    });

    const wCanvas = document.getElementById("wealth-canvas");
    const wTip = document.getElementById("wealth-tooltip");
    wCanvas.addEventListener("mousemove", (ev) => {
      const rows = BUNDLE.portfolio_monthly || [];
      if (!rows.length) return;
      const rect = wCanvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const idx = Math.round(((mx - 46) / Math.max(rect.width - 64, 1)) * rows.length);
      if (idx >= 0 && idx <= rows.length) {
        WEALTH_HOVER_IDX = idx;
        renderSimulator();
        const dtLabel = idx === 0 ? "2022-01" : String(rows[idx - 1].realization_dt).slice(0, 7);
        wTip.classList.remove("hidden");
        wTip.style.left = `${Math.min(rect.width - 210, Math.max(8, mx + 12))}px`;
        wTip.style.top = "12px";
        wTip.textContent = `${dtLabel} · mês ${idx} de ${rows.length}`;
      }
    });
    wCanvas.addEventListener("mouseleave", () => {
      WEALTH_HOVER_IDX = null;
      wTip.classList.add("hidden");
      renderSimulator();
    });

    const tbody = document.getElementById("portfolio-tbody");
    tbody.innerHTML = (BUNDLE.portfolio_summary || [])
      .map(
        (p) => `
        <tr style="${p.policy === "cvar_venn_abers_lagged" || p.policy === "ml_watchlist_cutoff" ? "background:#f1efe8" : ""}">
          <td><strong style="color:#1b2433">${POLICY_NAMES[p.policy] || p.policy}</strong></td>
          <td class="num">${fmtPct(p.cumulative_return_pct, 1)}</td>
          <td class="num"><strong style="color:#1b2433">${fmtNum(p.pct_of_cdi, 1)}%</strong></td>
          <td class="num">${fmtPct(p.cagr_aa_pct, 2)}</td>
          <td class="num">${p.excess_over_cdi_aa_pp >= 0 ? "+" : ""}${fmtNum(p.excess_over_cdi_aa_pp, 2)} pp</td>
          <td class="num">${fmtPct(p.volatility_aa_pct, 2)}</td>
          <td class="num">${fmtNum(p.sharpe_ratio, 2)}</td>
          <td class="num">${fmtNum(p.sortino_ratio, 2)}</td>
          <td class="num">${fmtPct(p.max_drawdown_pct, 2)}</td>
          <td class="num">${fmtPct(p.cvar_95_monthly_pct, 2)}</td>
          <td class="num">${p.severe_monthly_shocks_suffered}</td>
        </tr>`
      )
      .join("");
  }

  function renderSimulator() {
    const tau = parseFloat(document.getElementById("sl-sim-tau").value);
    const costRatio = parseInt(document.getElementById("sl-sim-cost").value, 10);
    const showFrictionless = document.getElementById("chk-show-frictionless").checked;

    document.getElementById("lbl-sim-tau").textContent = fmtPct(tau * 100, 1);
    document.getElementById("lbl-sim-cost").textContent = `${costRatio}×`;

    const frontier = BUNDLE.decision_frontier || [];
    if (!frontier.length) return;

    const cur = frontier.reduce((best, r) =>
      Math.abs(r.threshold - tau) < Math.abs(best.threshold - tau) ? r : best
    );
    const opt = frontier.reduce((best, r) =>
      r.fn * costRatio + r.fp < best.fn * costRatio + best.fp ? r : best
    );

    document.getElementById("sim-confusion-grid").innerHTML = `
      <div class="conf-box"><span>Quebras evitadas</span><strong>${fmtInt(cur.tp)}</strong></div>
      <div class="conf-box"><span>Alarmes falsos</span><strong style="color:#4a5160">${fmtInt(cur.fp)}</strong></div>
      <div class="conf-box"><span>Quebras não evitadas</span><strong style="color:#7a7f88">${fmtInt(cur.fn)}</strong></div>
      <div class="conf-box"><span>Recall · % do CDI</span><strong>${fmtPct(cur.recall * 100, 1)} · ${fmtNum(cur.pct_of_cdi, 0)}%</strong></div>
    `;

    document.getElementById("sim-optimal-callout").innerHTML =
      `Com uma quebra custando ${costRatio} vezes um alarme falso, o melhor corte é <b>${fmtPct(opt.threshold * 100, 1)}</b>: ` +
      `exclui ${fmtPct(opt.flagged_rate * 100, 1)} dos fundos e evita <b>${fmtPct(opt.recall * 100, 1)}</b> das quebras, ` +
      `com retorno de <b>${fmtNum(opt.pct_of_cdi, 1)}% do CDI</b> (Sharpe ${fmtNum(opt.sharpe_ratio, 2)}).`;

    drawWealthCanvas(showFrictionless);
  }

  function drawWealthCanvas(showFrictionless) {
    const canvas = document.getElementById("wealth-canvas");
    const { ctx, w, h } = setupHiDPICanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const rows = BUNDLE.portfolio_monthly || [];
    if (!rows.length) return;

    // Pure Monochrome Hierarchy differentiated by stroke weight & dash pattern
    const seriesDefs = [
      { key: "cdi_ret_pct", label: "100% CDI", color: "#a9acb2", dash: [2, 3], lw: 1.4, glow: false },
      { key: "equal_weight_lagged", label: "Peso igual", color: "#7a7f88", dash: [5, 4], lw: 1.4, glow: false },
      { key: "traditional_committee_lagged", label: "Regra de comitê", color: "#4a5160", dash: [], lw: 1.5, glow: false },
      { key: "yield_chasing_lagged", label: "Maiores retornos", color: "#4a5160", dash: [7, 3, 2, 3], lw: 1.6, glow: false },
      { key: "cvar_venn_abers_lagged", label: "Modelo (t+2)", color: "#1f3a5f", dash: [], lw: 2.5, glow: true },
    ];
    if (showFrictionless) {
      seriesDefs.push({
        key: "cvar_venn_abers_frictionless_t1",
        label: "Sem atraso de resgate (t+1)",
        color: "#1f3a5f",
        dash: [3, 3],
        lw: 1.5,
        glow: false,
      });
    }

    const curves = seriesDefs.map((s) => {
      let acc = 100.0;
      const pts = [100.0];
      rows.forEach((r) => {
        acc *= 1.0 + (Number(r[s.key]) || 0) / 100.0;
        pts.push(acc);
      });
      return { ...s, pts };
    });

    const yMin = Math.min(...curves.flatMap((c) => c.pts)) * 0.98;
    const yMax = Math.max(...curves.flatMap((c) => c.pts)) * 1.03;
    const pad = { l: 46, r: 18, t: 18, b: 28 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    const xAt = (i) => pad.l + (i / rows.length) * pw;
    const yAt = (v) => pad.t + ph - ((v - yMin) / Math.max(yMax - yMin, 1)) * ph;

    ctx.strokeStyle = "rgba(27,36,51,0.08)";
    ctx.fillStyle = "#7a7f88";
    ctx.font = "11px 'IBM Plex Sans', monospace";
    for (let g = 0; g <= 4; g++) {
      const val = yMin + ((yMax - yMin) * g) / 4;
      const yy = yAt(val);
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(w - pad.r, yy);
      ctx.stroke();
      ctx.fillText(val.toFixed(0), 6, yy + 3);
    }

    curves.forEach((c) => {
      ctx.save();
      if (c.glow) {
        ctx.shadowColor = "rgba(0,0,0,0)";
        ctx.shadowBlur = 8;
      }
      ctx.setLineDash(c.dash);
      ctx.strokeStyle = c.color;
      ctx.lineWidth = c.lw;
      ctx.beginPath();
      c.pts.forEach((v, i) => {
        if (i === 0) ctx.moveTo(xAt(i), yAt(v));
        else ctx.lineTo(xAt(i), yAt(v));
      });
      ctx.stroke();
      ctx.restore();
    });

    if (WEALTH_HOVER_IDX !== null) {
      const hx = xAt(WEALTH_HOVER_IDX);
      ctx.strokeStyle = "rgba(27,36,51,0.6)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, pad.t);
      ctx.lineTo(hx, h - pad.b);
      ctx.stroke();
    }

    let ly = pad.t + 10;
    curves.forEach((c) => {
      ctx.save();
      ctx.strokeStyle = c.color;
      ctx.lineWidth = c.lw;
      ctx.setLineDash(c.dash);
      ctx.beginPath();
      ctx.moveTo(pad.l + 10, ly - 4);
      ctx.lineTo(pad.l + 28, ly - 4);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = c.glow ? "#1f3a5f" : "#4a5160";
      ctx.font = `${c.glow ? "600" : "400"} 11px 'IBM Plex Sans', monospace`;
      const idxVal = WEALTH_HOVER_IDX !== null ? WEALTH_HOVER_IDX : c.pts.length - 1;
      ctx.fillText(`${c.label}: ${c.pts[idxVal].toFixed(1)}`, pad.l + 34, ly);
      ly += 15;
    });

    ctx.fillStyle = "#7a7f88";
    ctx.font = "11px 'IBM Plex Sans', monospace";
    ctx.fillText(String(rows[0].realization_dt).slice(0, 7), pad.l, h - 8);
    ctx.fillText(String(rows[rows.length - 1].realization_dt).slice(0, 7), w - pad.r - 50, h - 8);
  }

  /* ================= TAB 04: SCIENTIFIC TELEMETRY ================= */
  function initScienceTab() {
    document.getElementById("regime-select").addEventListener("change", renderBenchmarkTable);
    renderBenchmarkTable();
    renderForensicSection();
    renderDriftList();
  }

  function renderBenchmarkTable() {
    const mode = document.getElementById("regime-select").value;
    const rows =
      mode === "pooled"
        ? [...(BUNDLE.benchmarks_h6 || [])]
        : (BUNDLE.folds_h6 || []).filter((r) => r.fold === mode);

    rows.sort((a, b) => b.pr_auc - a.pr_auc);
    document.getElementById("benchmark-tbody").innerHTML = rows
      .map(
        (r, i) => `
        <tr style="${i === 0 ? "background:var(--navy-wash);box-shadow:inset 3px 0 0 var(--navy)" : ""}">
          <td><strong style="color:${i === 0 ? "#1b2433" : "#4a5160"}">${MODEL_NAMES[r.model] || r.model}</strong></td>
          <td class="num"><strong style="color:#1b2433">${fmtNum(r.pr_auc, 4)}</strong></td>
          <td class="num">${fmtNum(r.roc_auc, 4)}</td>
          <td class="num">${fmtNum(r.brier, 4)}</td>
          <td class="num">${fmtNum(r.ece, 4)}</td>
          <td class="num">${fmtPct(r.recall_top5 * 100, 1)}</td>
          <td class="num">${fmtNum(r.lift_top5, 2)}×</td>
        </tr>`
      )
      .join("");
  }

  function drawAblationCanvas() {
    const canvas = document.getElementById("ablation-canvas");
    const { ctx, w, h } = setupHiDPICanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const prog = (BUNDLE.ablation_h6 || []).filter((r) => r.ablation_type === "progressive");
    if (!prog.length) return;

    const shortLabels = [
      "Atraso e prazos",
      "+ sinais de maquiagem",
      "+ estrutura e liquidez",
      "+ cedentes e setores",
      "+ rede de contágio",
      "+ macro em nível",
    ];
    const pad = { l: 192, r: 58, t: 12, b: 18 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const xMin = 0.25;
    const xMax = Math.max(...prog.map((p) => p.pr_auc)) * 1.05;
    const rowH = ph / prog.length;
    const hatch = createDiagonalHatchPattern(ctx, "rgba(27,36,51,0.5)");

    prog.forEach((p, i) => {
      const yy = pad.t + i * rowH + rowH * 0.2;
      const bh = rowH * 0.6;
      const bw = ((p.pr_auc - xMin) / (xMax - xMin)) * pw;
      const isMacroDrop = i === prog.length - 1 && p.pr_auc < prog[i - 1].pr_auc;

      ctx.fillStyle = isMacroDrop ? "#4a5160" : "#1b2433";
      ctx.font = "11px 'IBM Plex Sans', monospace";
      ctx.fillText(shortLabels[i] || p.configuration.slice(0, 26), 4, yy + bh * 0.72);

      if (isMacroDrop) {
        ctx.fillStyle = hatch;
        ctx.fillRect(pad.l, yy, Math.max(bw, 4), bh);
        ctx.strokeStyle = "#1f3a5f";
        ctx.strokeRect(pad.l, yy, Math.max(bw, 4), bh);
      } else {
        ctx.fillStyle = i === prog.length - 2 ? "#1f3a5f" : "#4a5160";
        ctx.fillRect(pad.l, yy, Math.max(bw, 4), bh);
      }

      ctx.fillStyle = "#1b2433";
      ctx.font = "600 11px 'IBM Plex Sans', monospace";
      ctx.fillText(p.pr_auc.toFixed(4), pad.l + bw + 7, yy + bh * 0.74);
    });
  }

  function renderForensicSection() {
    const f = BUNDLE.forensic || {};
    const h1 = f.h1_recompra_masking || {};
    const h2 = f.h2_scr_provision_gap || {};
    const h3 = f.h3_shared_cedente_contagion || {};

    document.getElementById("forensic-cards").innerHTML = `
      <div class="f-card">
        <h4>Recompra como maquiagem</h4>
        <p>Em fundos com atraso &lt;3%, giro forense multiplica quebras em <b>${fmtNum(h1.relative_risk_ratio, 2)}×</b> (${fmtPct((h1.distress_6m_rate_clean || 0) * 100, 1)} → <b>${fmtPct((h1.distress_6m_rate_high_masking || 0) * 100, 1)}</b>, p=${Number(h1.fisher_p_value || 0).toExponential(0)}).</p>
      </div>
      <div class="f-card">
        <h4>Déficit no SCR</h4>
        <p>Hiato de PDD &gt;2 pp frente ao Bacen eleva estresse em <b>${fmtNum(h2.relative_risk_ratio, 2)}×</b> (${fmtPct((h2.distress_6m_rate_adequate || 0) * 100, 1)} → <b>${fmtPct((h2.distress_6m_rate_underprovisioned || 0) * 100, 1)}</b>, p=${Number(h2.fisher_p_value || 0).toExponential(0)}).</p>
      </div>
      <div class="f-card">
        <h4>Contágio pelo cedente</h4>
        <p>Exposição a originador estressado em outro FIDC eleva risco em <b>${fmtNum(h3.relative_risk_ratio, 2)}×</b> (${fmtPct((h3.distress_6m_rate_healthy_peers || 0) * 100, 1)} → <b>${fmtPct((h3.distress_6m_rate_distressed_peers || 0) * 100, 1)}</b>).</p>
      </div>
    `;
  }

  function drawEventStudyAndReliability() {
    const ev = BUNDLE.event_study || [];
    if (ev.length) {
      const c1 = document.getElementById("event-study-canvas");
      const { ctx, w, h } = setupHiDPICanvas(c1);
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 38, r: 16, t: 16, b: 26 };
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;
      const yMax = Math.max(...ev.map((r) => Math.max(r.inad_90p_mean_pct, r.inad_early_1_60_mean_pct))) * 1.15;

      const xAt = (i) => pad.l + (i / (ev.length - 1)) * pw;
      const yAt = (v) => pad.t + ph - (Math.max(0, v) / yMax) * ph;

      const zeroIdx = ev.findIndex((r) => r.tau === 0);
      if (zeroIdx >= 0) {
        ctx.strokeStyle = "rgba(27,36,51,0.45)";
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(xAt(zeroIdx), pad.t);
        ctx.lineTo(xAt(zeroIdx), h - pad.b);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const plotEv = (key, scale, col, dash, lw) => {
        ctx.save();
        ctx.setLineDash(dash);
        ctx.strokeStyle = col;
        ctx.lineWidth = lw;
        ctx.beginPath();
        ev.forEach((r, i) => {
          const xx = xAt(i);
          const yy = yAt(r[key] * scale);
          if (i === 0) ctx.moveTo(xx, yy);
          else ctx.lineTo(xx, yy);
        });
        ctx.stroke();
        ctx.restore();
      };
      plotEv("recompra_subst_mean_pct", 2.5, "#7a7f88", [2, 3], 1.5);
      plotEv("inad_early_1_60_mean_pct", 1.0, "#d9902b", [5, 3], 1.8);
      plotEv("inad_90p_mean_pct", 1.0, "#8e2323", [], 2.3);

      ctx.fillStyle = "#7a7f88";
      ctx.font = "11px 'IBM Plex Sans', monospace";
      ctx.fillText("τ=-9M", pad.l, h - 7);
      if (zeroIdx >= 0) {
        ctx.fillStyle = "#1b2433";
        ctx.fillText("ruptura", xAt(zeroIdx) - 42, h - 7);
      }
      ctx.fillStyle = "#7a7f88";
      ctx.fillText("+3m", w - pad.r - 24, h - 7);
    }

    const cal = BUNDLE.calibration_curves || [];
    if (cal.length) {
      const c2 = document.getElementById("reliability-canvas");
      const { ctx, w, h } = setupHiDPICanvas(c2);
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 34, r: 12, t: 14, b: 26 };
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;
      const maxP = 0.70;
      const xAt = (v) => pad.l + (Math.min(v, maxP) / maxP) * pw;
      const yAt = (v) => pad.t + ph - (Math.min(v, maxP) / maxP) * ph;

      ctx.strokeStyle = "rgba(27,36,51,0.22)";
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(xAt(0), yAt(0));
      ctx.lineTo(xAt(maxP), yAt(maxP));
      ctx.stroke();
      ctx.setLineDash([]);

      const styles = {
        lgbm_raw: { col: "#a9acb2", dash: [4, 3], lw: 1.6 },
        venn_abers: { col: "#4a5160", dash: [], lw: 1.8 },
        hybrid_stack: { col: "#1f3a5f", dash: [], lw: 2.3 },
      };
      cal.forEach((c) => {
        const st = styles[c.key];
        if (!st) return;
        ctx.save();
        ctx.setLineDash(st.dash);
        ctx.strokeStyle = st.col;
        ctx.lineWidth = st.lw;
        ctx.beginPath();
        c.points.forEach((pt, i) => {
          if (i === 0) ctx.moveTo(xAt(pt.pred), yAt(pt.obs));
          else ctx.lineTo(xAt(pt.pred), yAt(pt.obs));
        });
        ctx.stroke();
        ctx.restore();
      });
      ctx.fillStyle = "#4a5160";
      ctx.font = "11px 'IBM Plex Sans', monospace";
      ctx.fillText(" tracejado: bruto · contínuo: calibrado", 8, h - 7);
    }
  }

  function renderDriftList() {
    const drift = [...(BUNDLE.drift_psi || [])].sort((a, b) => b.psi - a.psi).slice(0, 12);
    document.getElementById("drift-list").innerHTML = drift
      .map(
        (d) => `
        <div class="drift-row">
          <span style="color:#4a5160"><b style="color:#1b2433">${d.semester}</b> · ${d.label_ptbr.slice(0, 34)}</span>
          <span style="color:${d.psi >= 0.25 ? "#1b2433" : "#7a7f88"};font-weight:${d.psi >= 0.25 ? "600" : "400"}">PSI ${d.psi.toFixed(2)}</span>
        </div>`
      )
      .join("");
  }

  function initTabs() {
    document.querySelectorAll(".tab-btn[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => activateTab(btn.getAttribute("data-tab")));
    });
    window.addEventListener("resize", () => {
      if (SELECTED_FUND) drawFundTrajectoryCanvas(SELECTED_FUND);
      drawContagionGraph();
      renderSimulator();
      drawAblationCanvas();
      drawEventStudyAndReliability();
    });
  }

  async function boot() {
    // Build demo: nada de API ao vivo nem de pacote nominal, mesmo que existam no servidor.
    let resp = null;
    if (!IS_DEMO) {
      await checkLiveApi();
      // Uso local (fidc-risk serve): pacote nominal de data/private/; senão, o público anonimizado.
      resp = await fetch("data/private/observatory_bundle.json", { cache: "no-cache" }).catch(() => null);
    }
    // no-cache: revalida com o servidor, para que a atualização mensal do pacote apareça sem cache velho.
    if (!resp || !resp.ok) resp = await fetch("data/observatory_bundle.json", { cache: "no-cache" });
    BUNDLE = await resp.json();
    const banner = document.getElementById("demo-banner");
    if (banner && BUNDLE.meta && BUNDLE.meta.demo) banner.hidden = false;

    populateKPIs();
    initCommandPalette();
    initTabs();
    initWatchlistControls();
    renderWatchlistTable();
    initContagionGraph();
    initSimulator();
    initScienceTab();
  }

  boot().catch((err) => {
    console.error("Erro ao iniciar terminal:", err);
    document.getElementById("header-meta").textContent = `FALHA DE TELEMETRIA: ${err.message}`;
  });
})();
