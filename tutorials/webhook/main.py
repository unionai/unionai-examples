import os

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from union import UnionRemote

app = FastAPI()

TOKEN = os.getenv("WEBHOOK_API_KEY")
security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> HTTPAuthorizationCredentials:
    if credentials.credentials != TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return credentials


@app.post("/run-workflow/{project}/{domain}/{name}/{version}")
def read_current_user(
    project: str,
    domain: str,
    name: str,
    version: str,
    inputs: dict,
    # credentials: Annotated[HTTPAuthorizationCredentials, Depends(verify_token)],
):
    remote = UnionRemote(default_domain=domain, default_project=project)
    wf = remote.fetch_workflow(name=name, version=version)
    e = remote.execute(wf, project=project, domain=domain, inputs=inputs)
    return {"url": remote.generate_console_url(e)}
