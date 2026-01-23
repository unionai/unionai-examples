# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
#    "websockets",
# ]
# ///

"""A FastAPI app with WebSocket support."""

import pathlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
from datetime import UTC, datetime
import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(
    title="Flyte WebSocket Demo",
    description="A FastAPI app with WebSocket support",
    version="1.0.0",
)

# {{docs-fragment connection-manager}}
class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all active connections."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting: {e}")

manager = ConnectionManager()
# {{/docs-fragment connection-manager}}

# {{docs-fragment websocket-endpoint}}
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await manager.send_personal_message(
            json.dumps({
                "type": "system",
                "message": "Welcome! You are connected.",
                "timestamp": datetime.now(UTC).isoformat(),
            }),
            websocket,
        )
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            
            # Echo back to sender
            await manager.send_personal_message(
                json.dumps({
                    "type": "echo",
                    "message": f"Echo: {data}",
                    "timestamp": datetime.now(UTC).isoformat(),
                }),
                websocket,
            )
            
            # Broadcast to all clients
            await manager.broadcast(
                json.dumps({
                    "type": "broadcast",
                    "message": f"Broadcast: {data}",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "connections": len(manager.active_connections),
                })
            )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(
            json.dumps({
                "type": "system",
                "message": "A client disconnected",
                "connections": len(manager.active_connections),
            })
        )
# {{/docs-fragment websocket-endpoint}}

# {{docs-fragment env}}
env = FastAPIAppEnvironment(
    name="websocket-app",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
        "websockets",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
)
# {{/docs-fragment env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed websocket app: {app_deployment[0].summary_repr()}")
# {{/docs-fragment deploy}}
