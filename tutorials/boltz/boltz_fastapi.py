# %%writefile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
from typing import Optional, Dict, Any
from boltz.main import predict  # Ensure you are importing the correct function
from click.testing import CliRunner
import io
from pathlib import Path
import traceback
import tempfile
import subprocess
import asyncio

app = FastAPI()

def package_outputs(output_dir: str) -> bytes:
    import io
    import tarfile

    tar_buffer = io.BytesIO()
    parent_dir = Path(output_dir).parent

    cur_dir = os.getcwd()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        os.chdir(parent_dir)
        try: 
            tar.add(Path(output_dir).name, arcname=Path(output_dir).name)
        finally: 
            os.chdir(cur_dir)

    return tar_buffer.getvalue()

@app.post("/predict/")
async def predict_endpoint(
    yaml_file: UploadFile = File(...),
    msa_dir: Optional[UploadFile] = File(None),
    options: Optional[Dict[str, str]] = Form(None)
):
    yaml_path = f"/tmp/{yaml_file.filename}"
    with open(yaml_path, "wb") as buffer:
        shutil.copyfileobj(yaml_file.file, buffer)

    msa_dir_path = None
    if msa_dir and msa_dir.filename:
        msa_dir_path = f"/tmp/{msa_dir.filename}"
        os.makedirs(msa_dir_path, exist_ok=True)
        with open(msa_dir_path, "wb") as buffer:
            shutil.copyfileobj(msa_dir.file, buffer)

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as out_dir:
        # Call boltz.predict as a CLI tool
        try:
            print(f"Running predictions with options: {options} into directory: {out_dir}")
            # Convert options dictionary to key-value pairs
            options_list = [f"--{key}={value}" for key, value in (options or {}).items()]
            command = ["boltz", "predict", yaml_path, "--out_dir", out_dir, "--accelerator", "cpu"] + options_list
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(stderr.decode())
            
            print(stdout.decode())
            
            # Package the output directory
            tar_data = package_outputs(f"{out_dir}/boltz_results_{Path(yaml_path).with_suffix('').name}")
            
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    return StreamingResponse(io.BytesIO(tar_data), media_type="application/gzip", headers={"Content-Disposition": f"attachment; filename=boltz_results.tar.gz"})

# %%
