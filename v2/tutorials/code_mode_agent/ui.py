"""Single-page chat UI for the Code Mode analytics agent.

Served at ``/``. Talks to ``/api/chat`` (ask a question) and ``/api/tools``
(list the tools shown in the sidebar). Each answer comes back as an ordered list
of self-contained HTML *blocks* (metric cards, Chart.js charts, tables) that this
page injects and runs, plus the list of tools the generated code called.
"""

CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Code Mode Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root{ --accent:#0ea5e9; --accent-2:#0284c7; --ink:#0f172a; --bg1:#f6f8fb; --bg2:#eef2f7;
         --surface:#ffffff; --surface-2:#f1f5f9;
         --line:#e2e8f0; --muted:#64748b; }
  *{ box-sizing:border-box; }
  html,body{ height:100%; }
  body{ margin:0; color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
        background:radial-gradient(1200px 600px at 80% -10%,rgba(14,165,233,.10),transparent 60%),
                   linear-gradient(135deg,var(--bg1),var(--bg2) 60%,var(--bg1));
        height:100vh; display:flex; flex-direction:column; }
  header{ display:flex; align-items:center; justify-content:space-between; gap:12px;
          padding:14px 22px; border-bottom:1px solid var(--line); backdrop-filter:blur(6px); }
  .brand{ display:flex; align-items:center; gap:11px; font-weight:700; font-size:16px; }
  .brand .logo{ width:28px; height:28px; border-radius:8px; display:grid; place-items:center; color:#fff;
                font-weight:800; font-size:15px; background:linear-gradient(135deg,var(--accent),var(--accent-2));
                box-shadow:0 2px 12px rgba(14,165,233,.35); }
  .brand .sub{ color:var(--muted); font-weight:500; font-size:14px; }
  .badge{ font-size:11px; font-weight:600; color:var(--accent-2); background:rgba(14,165,233,.12);
          border:1px solid rgba(14,165,233,.28); padding:5px 10px; border-radius:999px; white-space:nowrap; }
  main{ flex:1; display:flex; min-height:0; }
  #log{ flex:1; overflow-y:auto; padding:24px clamp(16px,5vw,72px); display:flex; flex-direction:column; gap:16px; }
  aside{ width:280px; border-left:1px solid var(--line); padding:18px; overflow-y:auto; }
  aside h3{ font-size:11px; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); margin:0 0 6px; }
  aside .hint{ font-size:12px; color:var(--muted); line-height:1.5; margin:0 0 14px; }
  .tool{ background:var(--surface); border:1px solid var(--line); border-radius:11px; padding:11px 12px; margin-bottom:8px; }
  .tool .thead{ display:flex; align-items:center; gap:7px; }
  .tool code{ color:var(--accent-2); font-size:12.5px; }
  .tool p{ margin:6px 0 0; font-size:12px; color:var(--muted); line-height:1.45; }
  .tag{ font-size:9.5px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; padding:2px 6px; border-radius:5px; }
  .tag.task{ color:#fff; background:linear-gradient(135deg,var(--accent),var(--accent-2)); }
  .tag.local{ color:var(--muted); background:var(--surface-2); border:1px solid var(--line); }
  .msg{ max-width:82%; padding:14px 16px; border-radius:16px; line-height:1.5; }
  .user{ align-self:flex-end; color:#fff; border-bottom-right-radius:5px; font-weight:500;
         background:linear-gradient(135deg,var(--accent),var(--accent-2)); box-shadow:0 4px 18px rgba(14,165,233,.25); }
  .bot{ align-self:flex-start; background:var(--surface); border:1px solid var(--line);
        border-bottom-left-radius:5px; width:min(96%,880px); }
  .bot .summary{ font-size:15px; }
  .report{ margin-top:12px; }
  .report canvas{ background:rgba(14,165,233,.04); border-radius:8px; }
  .tools-used{ display:flex; align-items:center; flex-wrap:wrap; gap:6px; margin-top:14px; }
  .tools-used .tu-label{ font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); margin-right:2px; }
  .tchip{ display:inline-flex; align-items:center; gap:6px; font-size:12px; background:var(--surface-2);
          border:1px solid var(--line); border-radius:999px; padding:4px 10px; }
  .tchip code{ color:var(--ink); font-size:12px; }
  .tchip.durable{ border-color:rgba(14,165,233,.4); background:rgba(14,165,233,.10); }
  .tchip.durable code{ color:var(--accent-2); }
  .tchip .x{ color:var(--muted); }
  details{ margin-top:12px; border-top:1px solid var(--line); padding-top:10px; }
  summary{ cursor:pointer; color:var(--accent-2); font-size:13px; font-weight:600; }
  pre{ background:var(--surface-2); border:1px solid var(--line); color:#0f172a; border-radius:8px; padding:14px; overflow-x:auto; font-size:12.5px; line-height:1.5; margin:10px 0 0; }
  code{ font-family:"Fira Code",Consolas,monospace; }
  .err{ color:#dc2626; white-space:pre-wrap; }
  .runlink{ margin-top:12px; font-size:13px; }
  .runlink a{ color:var(--accent-2); text-decoration:none; }
  .runlink a:hover{ text-decoration:underline; }
  .thinking{ display:inline-flex; gap:5px; align-items:center; color:var(--muted); }
  .thinking .dot{ width:6px; height:6px; border-radius:50%; background:var(--accent); animation:blink 1.2s infinite; }
  .thinking .dot:nth-child(2){ animation-delay:.2s; }
  .thinking .dot:nth-child(3){ animation-delay:.4s; }
  @keyframes blink{ 0%,60%,100%{ opacity:.25; } 30%{ opacity:1; } }
  footer{ padding:14px clamp(16px,5vw,72px); border-top:1px solid var(--line); }
  #suggestions{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
  .chip{ font-size:13px; color:var(--ink); background:var(--surface); border:1px solid var(--line);
         padding:7px 12px; border-radius:999px; cursor:pointer; transition:border-color .15s; }
  .chip:hover{ border-color:var(--accent); }
  .chip.primary{ border-color:var(--accent); color:var(--accent-2); font-weight:600;
                 background:rgba(14,165,233,.06); }
  .row{ display:flex; gap:10px; }
  #q{ flex:1; background:var(--surface); border:1px solid var(--line); border-radius:12px; color:var(--ink);
      padding:12px 14px; font-size:15px; outline:none; }
  #q:focus{ border-color:var(--accent); }
  #send{ border:none; border-radius:12px; padding:0 22px; font-weight:600; color:#fff; cursor:pointer;
         background:linear-gradient(135deg,var(--accent),var(--accent-2)); }
  #send:disabled{ opacity:.5; cursor:default; }
</style>
</head>
<body>
<header>
  <div class="brand"><span class="logo">&#9624;</span> Code Mode Analytics <span class="sub">chat with your data</span></div>
  <span class="badge" title="This app and the query tasks it runs are on Union">Served on Union</span>
</header>

<main>
  <div id="log">
    <div class="msg bot"><div class="summary">Ask a question about the <b>orders</b> dataset. I write one Python
      program that runs in a sandbox and can only call the tools on the right. It queries the data, then assembles a
      short report of numbers, charts, and tables. The heavy <code>query</code> runs as a durable Flyte task.
      <div style="margin-top:12px;"><button id="previewBtn" class="chip primary">Preview the dataset</button></div></div></div>
  </div>
  <aside>
    <h3>Tools</h3>
    <p class="hint">The model can only call these. <span style="color:var(--accent-2)">task</span> tools run as durable
      Flyte tasks; <span style="color:var(--muted)">local</span> tools run in-process.</p>
    <div id="tools"></div>
  </aside>
</main>

<footer>
  <div id="suggestions">
    <button class="chip">Give me a 2024 revenue overview</button>
    <button class="chip">Which regions and categories drive revenue?</button>
    <button class="chip">Return rate by channel</button>
    <button class="chip">Top months by units sold</button>
  </div>
  <div class="row">
    <input id="q" placeholder="Ask about the orders data…" autocomplete="off"/>
    <button id="send">Ask</button>
  </div>
</footer>

<script>
const log = document.getElementById('log');
const q = document.getElementById('q');
const send = document.getElementById('send');
let history = [];
let toolMeta = {};   // name -> { durable, ... } from /api/tools

function bubble(cls, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerHTML = html;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

// Blocks arrive as inert HTML; instantiate each chart canvas from its data-config.
// Charts are keyed by element (no ids), so duplicate or cached HTML can't collide.
function renderCharts(el) {
  el.querySelectorAll('canvas.cm-chart').forEach((c) => {
    if (window.Chart && Chart.getChart && Chart.getChart(c)) return;  // already drawn
    try { new Chart(c, JSON.parse(c.dataset.config)); }
    catch (e) { console.error('chart render failed', e); }
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

// The "Ran: query ×3  create_chart" strip. Durable tools get a highlighted chip.
function renderToolsUsed(used) {
  if (!used || !used.length) return '';
  const chips = used.map((t) => {
    const durable = toolMeta[t.name] && toolMeta[t.name].durable;
    const times = t.count > 1 ? ' <span class="x">×' + t.count + '</span>' : '';
    return '<span class="tchip' + (durable ? ' durable' : '') + '"><code>' +
      escapeHtml(t.name) + '</code>' + times + '</span>';
  }).join('');
  return '<div class="tools-used"><span class="tu-label">Ran</span>' + chips + '</div>';
}

async function ask(text) {
  if (!text.trim()) return;
  send.disabled = true;
  bubble('user', escapeHtml(text));
  history.push({ role: 'user', content: text });
  const pending = bubble('bot',
    '<span class="thinking">Writing code and running it<span class="dot"></span>' +
    '<span class="dot"></span><span class="dot"></span></span>');
  let runUrl = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: history.slice(0, -1) }),
    });
    // The server streams newline-delimited JSON: a "run" message with the link arrives
    // as soon as the run is submitted, then a "result" message when it finishes.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '', streamDone = false;
    while (!streamDone) {
      const { value, done } = await reader.read();
      streamDone = done;
      buf += decoder.decode(value || new Uint8Array(), { stream: !streamDone });
      let nl;
      while ((nl = buf.indexOf('\\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const msg = JSON.parse(line);
        if (msg.type === 'run') {
          runUrl = msg.run_url || '';
          pending.innerHTML =
            '<span class="thinking">Running the analysis<span class="dot"></span>' +
            '<span class="dot"></span><span class="dot"></span></span>' +
            (runUrl ? runLink(runUrl) : '');
          log.scrollTop = log.scrollHeight;
        } else if (msg.type === 'result') {
          renderResult(pending, msg, runUrl);
          if (!msg.error) history.push({ role: 'assistant', content: msg.summary || '' });
        }
      }
    }
  } catch (e) {
    pending.innerHTML = '<div class="err">Request failed: ' + escapeHtml(String(e)) + '</div>';
  } finally {
    send.disabled = false;
    log.scrollTop = log.scrollHeight;
    q.focus();
  }
}

function renderResult(el, data, runUrl) {
  const url = data.run_url || runUrl || '';
  if (data.error) {
    el.innerHTML = '<div class="err">' + escapeHtml(data.error) + '</div>' +
      (url ? runLink(url) : '');
    return;
  }
  const blocks = (data.blocks || []).join('');
  el.innerHTML =
    '<div class="summary">' + escapeHtml(data.summary || '') + '</div>' +
    renderToolsUsed(data.tools_used) +
    '<div class="report">' + blocks + '</div>' +
    (data.code ? '<details><summary>Generated code</summary><pre><code>' +
      escapeHtml(data.code) + '</code></pre></details>' : '') +
    (url ? runLink(url) : '');
  renderCharts(el);
}

function runLink(url) {
  return '<div class="runlink"><a href="' + url + '" target="_blank" rel="noopener">' +
    'View this analysis run in the Union UI &#8599;</a></div>';
}

// Show a sample of the data before querying it. This hits /api/dataset, a cheap
// in-process peek (no run launched) — handy for demoing the data first.
async function showDataset() {
  const b = bubble('bot', '<span class="thinking">Loading the data<span class="dot"></span>' +
    '<span class="dot"></span><span class="dot"></span></span>');
  try {
    const data = await (await fetch('/api/dataset')).json();
    if (data.error) { b.innerHTML = '<div class="err">' + escapeHtml(data.error) + '</div>'; return; }
    b.innerHTML = '<div class="summary">' + escapeHtml(data.summary || '') + '</div>' +
      '<div class="report">' + (data.table || '') + '</div>';
    log.scrollTop = log.scrollHeight;
  } catch (e) {
    b.innerHTML = '<div class="err">Request failed: ' + escapeHtml(String(e)) + '</div>';
  }
}

send.onclick = () => { const t = q.value; q.value = ''; ask(t); };
q.onkeydown = (e) => { if (e.key === 'Enter') send.onclick(); };
document.querySelectorAll('#suggestions .chip').forEach((c) => {
  c.onclick = () => ask(c.textContent);
});
document.getElementById('previewBtn').onclick = showDataset;

// Populate the tools sidebar from the registry, and remember which are durable.
fetch('/api/tools').then((r) => r.json()).then((toolsList) => {
  toolsList.forEach((t) => { toolMeta[t.name] = t; });
  document.getElementById('tools').innerHTML = toolsList.map((t) =>
    '<div class="tool"><div class="thead"><code>' + escapeHtml(t.name) + '</code>' +
    '<span class="tag ' + (t.durable ? 'task">task' : 'local">local') + '</span></div>' +
    '<p>' + escapeHtml(t.description) + '</p></div>'
  ).join('');
}).catch(() => {});
</script>
</body>
</html>
"""
