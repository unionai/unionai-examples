# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
#    "websockets",
# ]
# ///

"""WebSocket patterns: echo, broadcast, streaming, and chat."""

import asyncio
import json
import random
from datetime import datetime, UTC
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import flyte
from flyte.app.extras import FastAPIAppEnvironment


app = FastAPI(
    title="WebSocket Patterns Demo",
    description="Demonstrates various WebSocket patterns",
    version="1.0.0",
)


# {{docs-fragment echo-server}}
@app.websocket("/echo")
async def echo(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass
# {{/docs-fragment echo-server}}


# Connection manager for broadcast
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


# {{docs-fragment broadcast-server}}
@app.websocket("/broadcast")
async def broadcast(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
# {{/docs-fragment broadcast-server}}


# {{docs-fragment streaming-server}}
@app.websocket("/stream")
async def stream_data(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Generate or fetch data
            data = {"timestamp": datetime.now(UTC).isoformat(), "value": random.random()}
            await websocket.send_json(data)
            await asyncio.sleep(1)  # Send update every second
    except WebSocketDisconnect:
        pass
# {{/docs-fragment streaming-server}}


# {{docs-fragment chat-room}}
class ChatRoom:
    def __init__(self, name: str):
        self.name = name
        self.connections: list[WebSocket] = []
    
    async def join(self, websocket: WebSocket):
        self.connections.append(websocket)
    
    async def leave(self, websocket: WebSocket):
        self.connections.remove(websocket)
    
    async def broadcast(self, message: str, sender: WebSocket):
        for connection in self.connections:
            if connection != sender:
                await connection.send_text(message)


rooms: dict[str, ChatRoom] = {}


@app.websocket("/chat/{room_name}")
async def chat(websocket: WebSocket, room_name: str):
    await websocket.accept()
    
    if room_name not in rooms:
        rooms[room_name] = ChatRoom(room_name)
    
    room = rooms[room_name]
    await room.join(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            await room.broadcast(data, websocket)
    except WebSocketDisconnect:
        await room.leave(websocket)
# {{/docs-fragment chat-room}}


env = FastAPIAppEnvironment(
    name="websocket-patterns",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
        "websockets",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed WebSocket patterns app: {app_deployment[0].summary_repr()}")
