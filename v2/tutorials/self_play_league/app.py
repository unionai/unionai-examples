# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "fastapi",
#    "uvicorn",
# ]
# ///
"""Play Connect Four against the league champion in your browser.

The app loads the champion from the latest successful ``league`` run, serves a one-page board,
and answers each move with the network's choice, its move probabilities, and its value
estimate for the position. A ``--smoke`` league run counts too: if it is the latest, the app
serves its champion. The app requires a Union.ai login by default.

    uv run app.py         # deploy the app and print its URL
"""

from contextlib import asynccontextmanager

import flyte
import flyte.app
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel

state: dict = {}
CHAMPION_PATH = "/root/champion.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    from connect4 import PolicyValueNet

    net = PolicyValueNet()
    checkpoint = torch.load(CHAMPION_PATH, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model"])
    state["net"], state["generation"] = net.eval(), checkpoint.get("generation")
    yield


app = FastAPI(title="Connect Four arena", lifespan=lifespan)

# {{docs-fragment app}}
arena = FastAPIAppEnvironment(
    name="connect4-arena",
    app=app,
    # A CPU-only torch is enough to pick a move, and much smaller than the CUDA build.
    image=flyte.Image.from_debian_base(name="connect4-arena")
    .with_pip_packages("torch==2.9.1", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("numpy<3", "fastapi", "uvicorn"),
    parameters=[
        # The champion published by the most recent successful league run.
        flyte.app.Parameter(
            name="champion",
            value=flyte.app.RunOutput(type="file", task_name="c4-league-driver.league"),
            mount=CHAMPION_PATH,
            download=True,
        ),
    ],
    # The network is imported when the app starts, so the code bundle lists it.
    include=["connect4.py"],
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=600),
    # Only users signed in to your Union.ai tenant can open the app. To let anyone with the URL
    # play, add requires_auth=False.
)
# {{/docs-fragment app}}


class Position(BaseModel):
    # 6 rows, top to bottom, of 7 cells: 0 empty, 1 the champion's stone, -1 yours.
    board: list[list[int]]


# {{docs-fragment move}}
@app.post("/move")
async def move(position: Position) -> dict:
    """The champion's move for a position where it is the champion's turn."""
    import torch
    import torch.nn.functional as F

    board = torch.tensor(position.board, dtype=torch.float32).unsqueeze(0)
    legal = board[:, 0, :] == 0
    with torch.no_grad():
        logits, value = state["net"](board, legal)
    probs = F.softmax(logits, 1)[0]
    return {"column": int(probs.argmax()), "probs": [round(float(p), 3) for p in probs], "value": round(float(value), 3)}
# {{/docs-fragment move}}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE.replace("{generation}", str(state.get("generation", "?")))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connect Four arena</title><style>
:root{--bg:#f9f9f7;--ink:#0b0b0b;--muted:#52514e;--board:#2a78d6;--you:#eda100;--ai:#e34948;--hole:#f9f9f7}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--ink:#f0efec;--muted:#c3c2b7;--board:#256abf;--hole:#0d0d0d}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,sans-serif;display:flex;justify-content:center}
main{max-width:520px;width:100%;padding:16px}h1{font-size:22px;margin:8px 0}p{color:var(--muted)}
#board{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;background:var(--board);padding:8px;border-radius:12px}
.cell{aspect-ratio:1;border-radius:50%;background:var(--hole);cursor:pointer}.you{background:var(--you)}.ai{background:var(--ai)}
#bars{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;height:40px;align-items:end;margin:8px 8px 0}
.bar{background:var(--ai);border-radius:3px 3px 0 0;opacity:.7}button{margin-top:12px;padding:8px 14px;border-radius:8px;border:1px solid var(--muted);background:none;color:var(--ink);cursor:pointer}
</style></head><body><main><h1>Play the league champion</h1>
<p>You are yellow and move first. The champion is generation {generation} of a self-play league. The bars show its move probabilities.</p>
<div id="bars"></div><div id="board"></div><p id="status">Your move.</p><button onclick="reset()">New game</button></main>
<script>
let B, over;
const el = document.getElementById("board"), bars = document.getElementById("bars"), status_ = document.getElementById("status");
function reset(){ B = Array.from({length:6}, () => Array(7).fill(0)); over = false; draw([]); status_.textContent = "Your move."; }
function draw(probs){
  el.innerHTML = ""; for (let r=0;r<6;r++) for (let c=0;c<7;c++){ const d=document.createElement("div");
    d.className = "cell" + (B[r][c]===-1?" you":B[r][c]===1?" ai":""); d.onclick=()=>play(c); el.appendChild(d); }
  bars.innerHTML = ""; for (let c=0;c<7;c++){ const b=document.createElement("div"); b.className="bar"; b.style.height=((probs[c]||0)*40)+"px"; bars.appendChild(b); }
}
function drop(c, who){ for (let r=5;r>=0;r--) if (B[r][c]===0){ B[r][c]=who; return true; } return false; }
function wins(who){ const d=[[0,1],[1,0],[1,1],[1,-1]];
  for (let r=0;r<6;r++) for (let c=0;c<7;c++) for (const [dr,dc] of d){ let k=0;
    while (k<4 && r+dr*k>=0 && r+dr*k<6 && c+dc*k>=0 && c+dc*k<7 && B[r+dr*k][c+dc*k]===who) k++; if (k===4) return true; }
  return false; }
const full = () => B[0].every(x => x !== 0);
async function play(c){
  if (over || !drop(c, -1)) return; draw([]);
  if (wins(-1)) { over = true; status_.textContent = "You win."; return; }
  if (full()) { over = true; status_.textContent = "Draw."; return; }
  status_.textContent = "Thinking...";
  const res = await (await fetch("move", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({board:B})})).json();
  drop(res.column, 1); draw(res.probs);
  if (wins(1)) { over = true; status_.textContent = "The champion wins."; return; }
  if (full()) { over = true; status_.textContent = "Draw."; return; }
  status_.textContent = `Your move. The champion rates its position ${res.value >= 0 ? "+" : ""}${res.value.toFixed(2)} (from -1, lost, to +1, won).`;
}
reset();
</script></body></html>"""


# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.serve(arena).url)
# {{/docs-fragment deploy}}
