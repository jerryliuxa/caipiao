/* 号彩智算 - 前端交互逻辑（原生 JS，无外部依赖） */
"use strict";

const $ = (s) => document.querySelector(s);
const fmt = (n) => String(n).padStart(2, "0");

const state = {
  game: "ssq",
  games: {},
  algos: [],
  algo: "ensemble",
  stats: null,
  detailOpen: new Set(), // 展开明细的注序号
  lastPredict: null,     // 最近一次预测结果（用于保存跟踪/导出）
  cmpSel: new Set(),     // 对比面板已选算法
};

/* ---------------- 基础 ---------------- */
async function api(path) {
  const resp = await fetch(path);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `请求失败(${resp.status})`);
  return data;
}

let toastTimer = null;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

function ball(n, color, small) {
  return `<span class="ball ${color}${small ? " sm" : ""}">${fmt(n)}</span>`;
}
function ballsHTML(numbers, color) {
  return numbers.map((n) => ball(n, color)).join("");
}

/* ---------------- 初始化 ---------------- */
async function init() {
  try {
    const [games, algos] = await Promise.all([api("/api/games"), api("/api/algorithms")]);
    state.games = games;
    state.algos = algos.algorithms;
    state.availability = algos.availability || {};
  } catch (e) {
    toast("接口加载失败: " + e.message);
    return;
  }
  $("#tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-game]");
    if (btn) selectGame(btn.dataset.game);
  });
  $("#sel-window").addEventListener("change", () => { loadStats(); });
  $("#btn-update").addEventListener("click", updateData);
  $("#btn-predict").addEventListener("click", predict);
  $("#btn-backtest").addEventListener("click", backtest);
  $("#btn-compare").addEventListener("click", toggleComparePicker);
  $("#btn-sim").addEventListener("click", runSimulation);
  $("#btn-track-refresh").addEventListener("click", loadTracking);
  renderAlgoCards();
  await selectGame("ssq");
}

function selectGame(g) {
  state.game = g;
  state.lastPredict = null;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.game === g));
  $("#predict-result").innerHTML = "";
  $("#backtest-result").innerHTML = "";
  $("#compare-result").innerHTML = "";
  return Promise.all([loadStats(), loadHistory(), loadTracking()]);
}

/* ---------------- 数据栏 / 统计 ---------------- */
async function loadStats() {
  const game = state.games[state.game];
  const window = $("#sel-window").value;
  try {
    const stats = await api(`/api/stats?game=${state.game}&window=${window}`);
    state.stats = stats;
    if (stats.empty) {
      $("#draw-info").innerHTML = `<span class="chip">暂无数据，请点击「更新数据」</span>`;
      $("#heat-areas").innerHTML = `<div class="empty-tip">暂无开奖数据，点击右上角「更新数据」从数据源拉取</div>`;
      return;
    }
    renderDrawBar(game, stats);
    renderHeatmaps(stats);
    drawAllCharts(stats);
    $("#stats-label").textContent = `（近 ${stats.window} 期，共 ${stats.total} 期）`;
  } catch (e) {
    toast("统计加载失败: " + e.message);
  }
}

function renderDrawBar(game, stats) {
  const l = stats.latest;
  const balls = game.pools
    .map((p) => ballsHTML(l.numbers[p.key], p.color))
    .join(`<span class="sep">|</span>`);
  $("#draw-info").innerHTML = `
    <span class="chip">${game.name} <b>${l.issue}</b> 期 <span class="dim">${l.date}</span></span>
    <span class="chip">${balls}</span>
    <span class="chip">下期 <b>${stats.next_issue}</b> <span class="dim">${stats.next_date ? "约 " + stats.next_date + " 开奖" : ""}</span></span>`;
}

function renderHeatmaps(stats) {
  const areas = Object.values(stats.pools)
    .map((p) => {
      const rgb = p.color === "red" ? "226,60,48" : "47,111,228";
      const items = [];
      for (let n = 1; n <= p.max; n++) {
        const heat = p.heat[n - 1];
        const alpha = (0.10 + 0.88 * heat).toFixed(2);
        const color = heat > 0.55 ? "#fff" : "#4a5680"; // 深色球配白字，浅色球配深字
        items.push(
          `<span class="ball ${p.color}" style="background: rgba(${rgb},${alpha}); color:${color}; box-shadow:none" title="${fmt(n)}：出现 ${p.freq[n - 1]} 次 · 遗漏 ${p.omission[n - 1]} 期">${fmt(n)}</span>`
        );
      }
      return `<div class="pool-block">
        <div class="pool-title">${p.name}（${p.count} 个）
          <span class="legend">冷 <i class="bar" style="background:linear-gradient(90deg,#eceff5, rgb(${rgb}))"></i> 热</span>
        </div>
        <div class="heat">${items.join("")}</div>
      </div>`;
    })
    .join("");
  $("#heat-areas").innerHTML = areas;
}

/* ---------------- 图表（Canvas） ---------------- */
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function drawBars(cv, labels, values, rgb, highlightCount = 0) {
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const padL = 26, padB = 34, padT = 8;
  const cw = w - padL - 4, ch = h - padT - padB;
  const max = Math.max(...values, 1);
  const n = values.length;
  const bw = cw / n;
  ctx.font = "10px sans-serif";
  // 网格与纵轴
  ctx.strokeStyle = "rgba(26,35,64,.08)";
  ctx.fillStyle = "#8a94ad";
  for (let g = 0; g <= 2; g++) {
    const y = padT + ch - (ch * g) / 2;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - 4, y); ctx.stroke();
    ctx.fillText(String(Math.round((max * g) / 2)), 4, y + 3);
  }
  for (let i = 0; i < n; i++) {
    const bh = (values[i] / max) * ch;
    const x = padL + i * bw + bw * 0.15, y = padT + ch - bh;
    const wBar = bw * 0.7;
    if (highlightCount && i >= n - highlightCount) {
      ctx.fillStyle = `rgba(${rgb},1)`;
    } else {
      ctx.fillStyle = `rgba(${rgb},${(0.40 + 0.55 * (values[i] / max)).toFixed(2)})`;
    }
    ctx.beginPath();
    ctx.roundRect(x, y, wBar, Math.max(bh, 1), 2);
    ctx.fill();
    // 横轴标签（稀疏，避免重叠）
    const step = Math.ceil(n / 12);
    if (i % step === 0) {
      ctx.fillStyle = "#8a94ad";
      ctx.fillText(labels[i], x - 2, h - 14);
    }
  }
}

function drawSumChart(cv, series, avg, p10, p90) {
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const padL = 34, padB = 18, padT = 8;
  const cw = w - padL - 6, ch = h - padT - padB;
  const lo = Math.min(...series, p10) - 2, hi = Math.max(...series, p90) + 2;
  const x = (i) => padL + (i / (series.length - 1)) * cw;
  const y = (v) => padT + ch - ((v - lo) / (hi - lo)) * ch;
  // p10~p90 区间带
  ctx.fillStyle = "rgba(53,101,224,.08)";
  ctx.fillRect(padL, y(p90), cw, y(p10) - y(p90));
  // 参考线
  ctx.strokeStyle = "rgba(26,35,64,.12)";
  ctx.fillStyle = "#8a94ad";
  ctx.font = "10px sans-serif";
  [p10, avg, p90].forEach((v) => {
    ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(w - 6, y(v)); ctx.stroke();
    ctx.fillText(String(v), 4, y(v) + 3);
  });
  // 曲线
  ctx.strokeStyle = "#e8930c";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  series.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();
}

function drawAllCharts(stats) {
  const main = Object.values(stats.pools)[0];
  const labels = Array.from({ length: main.max }, (_, i) => String(i + 1));
  const order = main.freq; // 下标即号码
  drawBars($("#chart-freq"), labels, order, "53,101,224");
  drawBars($("#chart-omission"), labels, main.omission, "232,147,12");
  drawSumChart($("#chart-sum"), stats.sums.series, stats.sums.avg, stats.sums.p10, stats.sums.p90);
  drawBars($("#chart-parity"), stats.parity.labels, stats.parity.values, "14,164,114");
  drawPairsHeatmap($("#chart-pairs"), stats.pairs);
}

function drawPairsHeatmap(cv, matrix) {
  if (!matrix) return;
  const n = matrix.length;
  const cell = Math.min((cv.clientWidth - 36) / n, 12);
  cv.style.height = Math.round(cell * n + 30) + "px"; // 正方形矩阵自适应高度
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const ox = Math.round((w - n * cell) / 2), oy = 26;
  let max = 1;
  for (const row of matrix) for (const v of row) if (v > max) max = v;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      const t = Math.pow(v / max, 0.6); // 非线性拉伸，低值也可见
      const x = ox + j * cell, y = oy + i * cell;
      ctx.fillStyle = v === 0 ? "#f2f4f9" : `rgba(53,101,224,${(0.08 + 0.9 * t).toFixed(3)})`;
      ctx.fillRect(x, y, Math.max(cell - 1, 1), Math.max(cell - 1, 1));
    }
  }
  ctx.fillStyle = "#8a94ad";
  ctx.font = "9px sans-serif";
  const step = Math.ceil(n / 14);
  for (let i = 0; i < n; i += step) {
    ctx.fillText(String(i + 1), ox + i * cell, oy - 6);
    ctx.fillText(String(i + 1), ox - 14, oy + i * cell + cell / 2 + 3);
  }
}

/* ---------------- 历史开奖 ---------------- */
async function loadHistory() {
  const game = state.games[state.game];
  try {
    const data = await api(`/api/history?game=${state.game}&limit=30`);
    $("#history-label").textContent = `（最近 30 期 / 共 ${data.total} 期）`;
    const weekdays = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
    const cards = data.draws
      .map((d, idx) => {
        let wd = "";
        if (d.date) {
          const [y, m, dd] = d.date.split("-").map(Number);
          wd = weekdays[new Date(y, m - 1, dd).getDay()];
        }
        const latest = idx === 0;
        const balls = game.pools
          .map((p) => d.numbers[p.key].map((n) => ball(n, p.color)).join(""))
          .join(`<span class="sep">|</span>`);
        return `<div class="hist-card${latest ? " latest" : ""}">
          <div class="meta"><span class="issue">${d.issue}</span>${latest ? '<span class="badge">最新</span>' : ""}<span class="date">${d.date} ${wd}</span></div>
          <div class="balls">${balls}</div>
        </div>`;
      })
      .join("");
    $("#history-grid").innerHTML = cards;
  } catch (e) {
    toast("历史数据加载失败: " + e.message);
  }
}

/* ---------------- 更新数据 ---------------- */
async function updateData() {
  const btn = $("#btn-update");
  btn.disabled = true; btn.textContent = "更新中…";
  try {
    const r = await api(`/api/update?game=${state.game}&limit=150`);
    const info = r.result[state.game];
    toast(`已拉取 ${info.fetched} 期，新增 ${info.added} 期` + (info.settled ? `，核对历史预测 ${info.settled} 注` : ""));
    await Promise.all([loadStats(), loadHistory(), loadTracking()]);
  } catch (e) {
    toast("更新失败: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "⟳ 更新数据";
  }
}

/* ---------------- 算法选择 ---------------- */
const CATEGORY_ORDER = ["综合推荐", "经典统计", "高级概率", "机器学习", "深度学习", "时序预测", "优化与组合"];

function renderAlgoCards() {
  const groups = CATEGORY_ORDER.filter((c) => state.algos.some((a) => a.category === c));
  $("#algo-cards").innerHTML = groups
    .map((cat) => {
      const cards = state.algos
        .filter((a) => a.category === cat)
        .map(
          (a) => `<div class="algo-card ${a.key === state.algo ? "active" : ""}" data-algo="${a.key}">
            <div class="name"><span class="dot"></span>${a.name}${a.key === "ensemble" ? '<span class="tag">基础</span>' : ""}${a.key === "stacking" ? '<span class="tag">推荐</span>' : ""}${a.heavy ? '<span class="tag heavy">numpy</span>' : ""}</div>
            <div class="desc">${a.desc}</div>
          </div>`
        )
        .join("");
      return `<div class="algo-group"><div class="algo-cat">${cat}<span>${state.algos.filter((a) => a.category === cat).length} 种</span></div><div class="algo-grid">${cards}</div></div>`;
    })
    .join("");
  if (state.availability && state.availability.sklearn === false) {
    $("#algo-cards").insertAdjacentHTML(
      "afterbegin",
      `<div class="dep-hint">未检测到 numpy / scikit-learn，机器学习与深度学习模型未启用 —— 运行 <code>pip install numpy scikit-learn</code> 后重启可解锁全部 ${state.availability.total} 个算法</div>`
    );
  }
  $("#algo-cards").addEventListener("click", (e) => {
    const card = e.target.closest(".algo-card");
    if (!card) return;
    state.algo = card.dataset.algo;
    document.querySelectorAll(".algo-card").forEach((c) => c.classList.toggle("active", c.dataset.algo === state.algo));
    renderAlgoParams();
    $("#predict-result").innerHTML = "";
  });
  renderAlgoParams();
}

function renderAlgoParams() {
  const algo = state.algos.find((a) => a.key === state.algo);
  if (!algo || !algo.params.length) {
    $("#algo-params").innerHTML = "";
    return;
  }
  $("#algo-params").innerHTML = algo.params
    .map(
      (p) => `<label>${p.label}
        <input type="range" data-key="${p.key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">
        <span class="pv" data-pv="${p.key}">${p.default}</span>
      </label>`
    )
    .join("");
  $("#algo-params").addEventListener("input", (e) => {
    const key = e.target.dataset.key;
    if (key) {
      const pv = document.querySelector(`[data-pv="${key}"]`);
      if (pv) pv.textContent = e.target.value;
    }
  });
}

function collectParams() {
  const params = {};
  document.querySelectorAll("#algo-params input[data-key]").forEach((inp) => {
    params[inp.dataset.key] = parseFloat(inp.value);
  });
  return params;
}

/* ---------------- 预测 ---------------- */
async function predict() {
  const btn = $("#btn-predict");
  btn.disabled = true; btn.textContent = "计算中…";
  try {
    const game = state.games[state.game];
    const window = $("#sel-window").value;
    const sets = $("#sel-sets").value;
    const params = encodeURIComponent(JSON.stringify(collectParams()));
    const nonce = Date.now() % 1e9;
    const r = await api(`/api/predict?game=${state.game}&algo=${state.algo}&window=${window}&sets=${sets}&nonce=${nonce}&params=${params}`);
    renderPredict(game, r);
    $("#backtest-result").innerHTML = "";
  } catch (e) {
    toast("预测失败: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "🎲 生成预测";
  }
}

function renderPredict(game, r) {
  state.lastPredict = r;
  const canTrack = state.stats && !state.stats.empty;
  const tickets = r.tickets
    .map((t, i) => {
      const balls = game.pools
        .map((p) => ballsHTML(t.numbers[p.key], p.color))
        .join(`<span class="sep">|</span>`);
      const open = state.detailOpen.has(i) ? "open" : "";
      const details = game.pools
        .map(
          (p) => t.detail[p.key]
            .map(
              (d) => `<span class="detail-item"><b class="${p.color === "red" ? "red-t" : "blue-t"}" style="color:${p.color === "red" ? "#ff7a6e" : "#6ea8ff"}">${fmt(d.n)}</b>
                热度 ${d.heat} · 出现 ${d.freq} 次 · 遗漏 ${d.omission} 期</span>`
            )
            .join("")
        )
        .join("");
      return `<div class="ticket ${open}" data-idx="${i}" style="cursor:pointer" title="点击展开/收起号码解读">
        <span class="no">第 ${i + 1} 注</span>
        <span class="balls">${balls}</span>
        <span class="score">综合热度 <b>${t.score}</b></span>
        <div class="detail-grid">${state.detailOpen.has(i) ? details : ""}</div>
      </div>`;
    })
    .join("");

  const ranking = Object.keys(r.ranking)
    .map((pk) => {
      const pool = game.pools.find((p) => p.key === pk);
      const maxScore = Math.max(...r.ranking[pk].map((x) => x.score));
      const rows = r.ranking[pk]
        .map(
          (x, i) => `<tr><td>${i + 1}</td><td><b>${fmt(x.n)}</b></td><td>${x.freq} 次</td><td>遗漏 ${x.omission}</td>
            <td><div class="scorebar ${pool.color}"><i style="width:${((x.score / maxScore) * 100).toFixed(0)}%"></i></div></td></tr>`
        )
        .join("");
      return `<div class="rank-cols"><table><thead><tr><th colspan="5">${pool.name} 评分 Top10</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    })
    .join("");

  $("#predict-result").innerHTML = `
    <div class="summary"><div class="head">🧠 ${r.algo_name} · 基于 ${r.based_on.count} 期（${r.based_on.from} ~ ${r.based_on.to}）</div>${r.summary}</div>
    <div class="tickets">${tickets}</div>
    <div class="pred-actions">
      <button class="btn sm" id="btn-track" ${canTrack ? "" : "disabled"}>💾 保存并跟踪（下期开奖自动核对）</button>
      <button class="btn sm" id="btn-export">⬇ 导出 CSV</button>
      <span class="hint-inline">跟踪记录将按算法长期累积，见下方「预测跟踪」</span>
    </div>
    <div class="rank-table"><div class="rank-wrap">${ranking}</div></div>`;

  $("#btn-track").addEventListener("click", trackSave);
  $("#btn-export").addEventListener("click", exportCSV);

  document.querySelectorAll("#predict-result .ticket").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = Number(el.dataset.idx);
      state.detailOpen.has(idx) ? state.detailOpen.delete(idx) : state.detailOpen.add(idx);
      renderPredict(game, r);
    });
  });
}

/* ---------------- 回测 ---------------- */
async function backtest() {
  const btn = $("#btn-backtest");
  btn.disabled = true; btn.textContent = "回测中…";
  try {
    const window = $("#sel-window").value;
    const params = encodeURIComponent(JSON.stringify(collectParams()));
    const r = await api(`/api/backtest?game=${state.game}&algo=${state.algo}&window=${window}&count=30&params=${params}`);
    renderBacktest(r);
  } catch (e) {
    toast("回测失败: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "📈 回测验证（近30期）";
  }
}

function renderBacktest(r) {
  const s = r.summary;
  const game = state.games[state.game];
  const lift = s.avg_main_hits / s.baseline_main_hits;
  const liftCls = lift >= 1.02 ? "good" : lift <= 0.98 ? "bad" : "";
  const rows = r.rows
    .slice()
    .reverse()
    .map((row) => {
      const predBalls = game.pools
        .map((p) => {
          const actual = new Set(row.actual[p.key]);
          return row.predicted[p.key]
            .map((n) => `<span class="ball sm ${p.color}" style="${actual.has(n) ? "" : "opacity:.32"}">${fmt(n)}</span>`)
            .join("");
        })
        .join(`<span class="sep">|</span>`);
      const actualBalls = game.pools
        .map((p) => row.actual[p.key].map((n) => ball(n, p.color, true)).join(""))
        .join(`<span class="sep">|</span>`);
      return `<tr><td><b>${row.issue}</b></td><td style="color:var(--muted)">${row.date}</td>
        <td><span class="balls">${predBalls}</span></td>
        <td><span class="balls">${actualBalls}</span></td>
        <td class="hit">${row.main_hits} + ${row.sub_hits}</td></tr>`;
    })
    .join("");

  $("#backtest-result").innerHTML = `
    <div class="summary"><div class="head">📈 ${r.algo_name} · 滚动回测（近 ${s.count} 期，每期仅用其之前 ${r.window} 期数据预测一注）</div>
    括号内为随机选号的理论平均值。命中列为「${game.pools[0].name}命中 + ${game.pools[1].name}命中」；预测号中<b style="color:var(--gold)">高亮</b>为命中号、<span style="opacity:.5">半透明</span>为未中。</div>
    <div class="bt-summary">
      <div class="bt-item"><div class="k">主池平均命中（随机 ${s.baseline_main_hits}）</div><div class="v ${liftCls}">${s.avg_main_hits}</div></div>
      <div class="bt-item"><div class="k">主池命中 ≥ ${s.main_hit_threshold} 个占比</div><div class="v">${(s.main_hit_rate_ge * 100).toFixed(0)}%</div></div>
      <div class="bt-item"><div class="k">副池平均命中（随机 ${s.baseline_sub_hits}）</div><div class="v">${s.avg_sub_hits}</div></div>
      <div class="bt-item"><div class="k">副池全中占比</div><div class="v">${(s.sub_full_rate * 100).toFixed(0)}%</div></div>
    </div>
    <div class="table-wrap" style="margin-top:14px">
      <table class="hist">
        <thead><tr><th>期号</th><th>开奖日</th><th>预测号码</th><th>实际开奖</th><th>命中</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ---------------- 保存跟踪 / 导出 ---------------- */
async function trackSave() {
  if (!state.lastPredict) return;
  try {
    const body = {
      game: state.game,
      algo: state.lastPredict.algo,
      params: collectParams(),
      issue_target: state.stats ? state.stats.next_issue : "",
      tickets: state.lastPredict.tickets.map((t) => t.numbers),
    };
    const resp = await fetch("/api/track", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "保存失败");
    toast(`已保存 ${data.ids.length} 注，目标期号 ${body.issue_target}，开奖后自动核对`);
    loadTracking();
  } catch (e) {
    toast("保存失败: " + e.message);
  }
}

function exportCSV() {
  const r = state.lastPredict;
  if (!r) return;
  const game = state.games[state.game];
  const lines = ["注号,池,号码,综合热度"];
  r.tickets.forEach((t, i) => {
    game.pools.forEach((p) => {
      lines.push(`${i + 1},${p.name},"${t.numbers[p.key].map(fmt).join(" ")}",${t.score}`);
    });
  });
  lines.push("");
  lines.push(`算法,${r.algo_name}`);
  lines.push(`基于,${r.based_on.count}期(${r.based_on.from}~${r.based_on.to})`);
  const blob = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `预测_${r.game}_${r.algo}_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
  toast("CSV 已导出");
}

/* ---------------- 算法对比 ---------------- */
function toggleComparePicker() {
  const el = $("#cmp-picker");
  if (el.classList.contains("open")) {
    el.classList.remove("open");
    el.innerHTML = "";
    return;
  }
  el.classList.add("open");
  el.innerHTML = `
    <div class="cmp-head">勾选 2~6 个算法并排对比（各出一注 + 快速回测 + 跟踪战绩）</div>
    <div class="cmp-chips">${state.algos
      .map((a) => `<button class="chip-toggle ${state.cmpSel.has(a.key) ? "on" : ""}" data-algo="${a.key}">${a.name}${a.heavy ? " ᴺ" : ""}</button>`)
      .join("")}</div>
    <div class="controls">
      <label class="inline">快速回测期数 <select id="cmp-bt"><option>0</option><option value="8" selected>8</option><option>15</option></select></label>
      <button class="btn primary sm2" id="btn-run-compare">⚖️ 运行对比</button>
      <span class="hint-inline">含深度学习算法时首次对比较慢（逐算法重新训练）</span>
    </div>`;
  el.querySelectorAll(".chip-toggle").forEach((chip) => {
    chip.addEventListener("click", () => {
      const k = chip.dataset.algo;
      if (state.cmpSel.has(k)) state.cmpSel.delete(k);
      else if (state.cmpSel.size >= 6) { toast("最多同时对比 6 个算法"); return; }
      else state.cmpSel.add(k);
      chip.classList.toggle("on", state.cmpSel.has(k));
    });
  });
  $("#btn-run-compare").addEventListener("click", runCompare);
}

async function runCompare() {
  if (!state.cmpSel.size) { toast("请先勾选要对比的算法"); return; }
  const btn = $("#btn-run-compare");
  btn.disabled = true; btn.textContent = "对比中…";
  try {
    const algos = [...state.cmpSel].join(",");
    const bt = $("#cmp-bt").value;
    const r = await api(`/api/compare?game=${state.game}&algos=${algos}&window=${$("#sel-window").value}&backtest=${bt}`);
    renderCompare(r);
  } catch (e) {
    toast("对比失败: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "⚖️ 运行对比";
  }
}

function renderCompare(r) {
  const game = state.games[r.game];
  const best = Math.max(...r.results.map((x) => (x.bt ? x.bt.avg_main_hits : -1)));
  const rows = r.results
    .map((x) => {
      const balls = game.pools
        .map((p) => x.ticket.numbers[p.key].map((n) => ball(n, p.color, true)).join(""))
        .join(`<span class="sep">|</span>`);
      const btCell = x.bt
        ? `<b class="${x.bt.avg_main_hits >= x.bt.baseline_main_hits ? "good-t" : "dim-t"}">${x.bt.avg_main_hits}</b> <span class="dim-t">(基线 ${x.bt.baseline_main_hits})</span>`
        : '<span class="dim-t">—</span>';
      const tk = r.tracking[x.algo];
      const trackCell = tk && tk.settled
        ? `${tk.avg_main} / ${tk.settled} 注${tk.sub_hit_count ? ` · 副中 ${tk.sub_hit_count}` : ""}`
        : '<span class="dim-t">暂无跟踪</span>';
      return `<tr class="${x.bt && x.bt.avg_main_hits === best && best > 0 ? "best" : ""}">
        <td><b>${x.name}</b><div class="dim-t">${x.category}</div></td>
        <td><span class="balls">${balls}</span></td>
        <td>${x.ticket.score}</td>
        <td>${btCell}</td>
        <td>${trackCell}</td></tr>`;
    })
    .join("");
  $("#compare-result").innerHTML = `
    <div class="summary"><div class="head">⚖️ 算法对比 · 各出一注（基于 ${r.window} 期，回测 ${r.backtest_count} 期）</div>
    加粗行为回测主池平均命中最高者。回测极短，仅供粗略参考；「跟踪战绩」才是长期真实表现。</div>
    <div class="table-wrap"><table class="hist">
      <thead><tr><th>算法</th><th>一注号码</th><th>综合热度</th><th>回测均命中</th><th>跟踪均命中</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

/* ---------------- 预测跟踪 ---------------- */
async function loadTracking() {
  try {
    const r = await api(`/api/tracking?game=${state.game}&limit=60`);
    const body = $("#tracking-body");
    if (!r.stats.length && !r.rows.length) {
      body.innerHTML = `<div class="empty-tip">暂无跟踪记录，先去生成一次预测并保存</div>`;
      $("#tracking-label").textContent = "";
      return;
    }
    const statRows = r.stats
      .map((s) => {
        const name = (state.algos.find((a) => a.key === s.algo) || {}).name || s.algo;
        return `<tr><td><b>${name}</b></td><td>${s.total}</td><td>${s.settled}</td>
          <td>${s.avg_main ?? "—"}</td><td>${s.avg_sub ?? "—"}</td><td>${s.sub_hit_count}</td></tr>`;
      })
      .join("");
    const rowHtml = r.rows
      .map((row) => {
        const name = (state.algos.find((a) => a.key === row.algo) || {}).name || row.algo;
        const balls = state.games[state.game].pools
          .map((p) => row.numbers[p.key].map((n) => ball(n, p.color, true)).join(""))
          .join(`<span class="sep">|</span>`);
        const status = row.settled
          ? `<span class="hit">命中 ${row.main_hits} + ${row.sub_hits}</span>`
          : '<span class="dim-t">未开奖</span>';
        return `<tr><td><b>${row.issue_target}</b></td><td>${name}</td>
          <td><span class="balls">${balls}</span></td><td>${status}</td><td class="dim-t">${row.created_at}</td></tr>`;
      })
      .join("");
    const totalSettled = r.stats.reduce((a, s) => a + s.settled, 0);
    $("#tracking-label").textContent = `（${state.games[state.game].name} · 累计 ${r.stats.reduce((a, s) => a + s.total, 0)} 注，已开奖 ${totalSettled} 注）`;
    body.innerHTML = `
      <h3 class="sub-h">算法战绩</h3>
      <div class="table-wrap"><table class="hist">
        <thead><tr><th>算法</th><th>累计注数</th><th>已开奖</th><th>主池均命中</th><th>副池均命中</th><th>副池命中次数</th></tr></thead>
        <tbody>${statRows}</tbody></table></div>
      <h3 class="sub-h">最近记录</h3>
      <div class="table-wrap"><table class="hist">
        <thead><tr><th>目标期号</th><th>算法</th><th>号码</th><th>状态</th><th>保存时间</th></tr></thead>
        <tbody>${rowHtml}</tbody></table></div>`;
  } catch (e) {
    toast("跟踪加载失败: " + e.message);
  }
}

/* ---------------- 模拟购彩计算器 ---------------- */
const PRIZE = {
  ssq: { price: 2, currency: "¥", tiers: [
    { m: 6, s: 1, p: 7000000, name: "一等" }, { m: 6, s: 0, p: 200000, name: "二等" },
    { m: 5, s: 1, p: 3000 }, { m: 5, s: 0, p: 200 }, { m: 4, s: 1, p: 200 },
    { m: 4, s: 0, p: 10 }, { m: 3, s: 1, p: 10 },
    { m: 2, s: 1, p: 5 }, { m: 1, s: 1, p: 5 }, { m: 0, s: 1, p: 5 },
  ] },
  dlt: { price: 2, currency: "¥", tiers: [
    { m: 5, s: 2, p: 10000000, name: "一等" }, { m: 5, s: 1, p: 500000, name: "二等" },
    { m: 5, s: 0, p: 10000 }, { m: 4, s: 2, p: 3000 }, { m: 4, s: 1, p: 300 },
    { m: 3, s: 2, p: 200 }, { m: 4, s: 0, p: 100 },
    { m: 3, s: 1, p: 15 }, { m: 2, s: 2, p: 15 },
    { m: 3, s: 0, p: 5 }, { m: 1, s: 2, p: 5 }, { m: 2, s: 1, p: 5 }, { m: 0, s: 2, p: 5 },
  ] },
  pb: { price: 2, currency: "$", tiers: [
    { m: 5, s: 1, p: 20000000, name: "头奖(最低)" }, { m: 5, s: 0, p: 1000000 },
    { m: 4, s: 1, p: 50000 }, { m: 4, s: 0, p: 100 }, { m: 3, s: 1, p: 100 },
    { m: 3, s: 0, p: 7 }, { m: 2, s: 1, p: 7 }, { m: 1, s: 1, p: 4 }, { m: 0, s: 1, p: 4 },
  ] },
  mm: { price: 5, currency: "$", tiers: [
    { m: 5, s: 1, p: 50000000, name: "头奖(最低)" }, { m: 5, s: 0, p: 10000000 },
    { m: 4, s: 1, p: 100000 }, { m: 4, s: 0, p: 1000 }, { m: 3, s: 1, p: 1000 },
    { m: 3, s: 0, p: 50 }, { m: 2, s: 1, p: 50 }, { m: 1, s: 1, p: 25 }, { m: 0, s: 1, p: 10 },
  ] },
};

function _sample(arr, k) {
  const a = arr.slice();
  for (let i = a.length - 1; i > a.length - 1 - k && i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(a.length - k);
}

function _prizeOf(game, mh, sh) {
  for (const t of PRIZE[game].tiers) {
    if (t.m === mh && t.s === sh) return t.p;
  }
  return 0;
}

function runSimulation() {
  const game = state.game;
  const cfg = state.games[game];
  const pools = cfg.pools;
  if (!PRIZE[game]) { toast("该游戏暂无奖级表"); return; }
  const M = parseInt($("#sim-bets").value, 10);
  const N = parseInt($("#sim-periods").value, 10);
  const price = PRIZE[game].price;
  const cur = PRIZE[game].currency || "¥";
  const mainArr = Array.from({ length: pools[0].max_number }, (_, i) => i + 1);
  const subArr = Array.from({ length: pools[1].max_number }, (_, i) => i + 1);
  const K = pools[0].count, SK = pools[1].count;

  let cost = 0, win = 0, jackpot = 0, best = 0, hitAny = 0;
  const costSeries = [], winSeries = [];
  for (let period = 1; period <= N; period++) {
    const dMain = new Set(_sample(mainArr, K));
    const dSub = new Set(_sample(subArr, SK));
    for (let b = 0; b < M; b++) {
      cost += price;
      const tMain = new Set(_sample(mainArr, K));
      const tSub = new Set(_sample(subArr, SK));
      let mh = 0, sh = 0;
      dMain.forEach((x) => { if (tMain.has(x)) mh++; });
      dSub.forEach((x) => { if (tSub.has(x)) sh++; });
      const prize = _prizeOf(game, mh, sh);
      win += prize;
      if (prize > 0) hitAny++;
      if (prize > best) best = prize;
      if (mh === K && sh === SK) jackpot++;
    }
    costSeries.push(cost);
    winSeries.push(win);
  }
  const ret = cost ? (win / cost * 100) : 0;
  $("#sim-result").innerHTML = `
    <div class="bt-summary">
      <div class="bt-item"><div class="k">模拟规模</div><div class="v">${N} 期 × ${M} 注</div></div>
      <div class="bt-item"><div class="k">总投入</div><div class="v">${cur}${cost.toLocaleString()}</div></div>
      <div class="bt-item"><div class="k">总收回</div><div class="v">${cur}${win.toLocaleString()}</div></div>
      <div class="bt-item"><div class="k">回报率</div><div class="v ${ret >= 100 ? "good" : "bad"}">${ret.toFixed(1)}%<small>（长期理论约 50%）</small></div></div>
      <div class="bt-item"><div class="k">中奖次数 / 头奖</div><div class="v">${hitAny} / ${jackpot}</div></div>
      <div class="bt-item"><div class="k">单次最大中奖</div><div class="v">${cur}${best.toLocaleString()}</div></div>
    </div>
    <div class="chart-card" style="margin-top:12px"><h3>累计投入 vs 累计收回</h3><canvas id="chart-sim" style="height:220px"></canvas></div>`;
  drawSimChart($("#chart-sim"), costSeries, winSeries, cur);
}

function drawSimChart(cv, cost, win, cur = "¥") {
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const padL = 56, padB = 18, padT = 10;
  const cw = w - padL - 8, ch = h - padT - padB;
  const hi = Math.max(...cost, ...win, 1);
  const x = (i) => padL + (i / (cost.length - 1)) * cw;
  const y = (v) => padT + ch - (v / hi) * ch;
  ctx.strokeStyle = "rgba(26,35,64,.10)";
  ctx.fillStyle = "#8a94ad";
  ctx.font = "10px sans-serif";
  for (let g = 0; g <= 3; g++) {
    const vv = (hi * g) / 3;
    ctx.beginPath(); ctx.moveTo(padL, y(vv)); ctx.lineTo(w - 6, y(vv)); ctx.stroke();
    ctx.fillText(cur + Math.round(vv / 1000) + "k", 4, y(vv) + 3);
  }
  const line = (arr, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.8;
    ctx.beginPath();
    arr.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
  };
  line(cost, "#e8930c");
  line(win, "#0ea472");
  ctx.fillStyle = "#e8930c"; ctx.fillRect(w - 150, 10, 10, 3); ctx.fillText("累计投入", w - 134, 14);
  ctx.fillStyle = "#0ea472"; ctx.fillRect(w - 150, 26, 10, 3); ctx.fillText("累计收回", w - 134, 30);
}

/* 启动 */
window.addEventListener("resize", () => { if (state.stats && !state.stats.empty) drawAllCharts(state.stats); });
init();
