# # Deploy Marimo Notebooks as WASM Applications
#
# In this example, we use Union Serving to deploy a Marimo notebook as a WebAssembly (WASM)
# application. This allows you to run interactive notebooks directly in the browser without
# requiring a Python runtime on the server side.

# {{run-on-union}}

# ## Overview
#
# This tutorial demonstrates how to:
# 1. Create a Marimo notebook with interactive content
# 2. Export the notebook as a WASM component
# 3. Deploy it using Union Serving
# 4. Access the interactive notebook in your browser
#
# Marimo notebooks exported as WASM run entirely in the browser, making them perfect
# for sharing interactive data visualizations and analyses without server-side dependencies.

# ## Defining the Application Configuration
#
# First, we define the image spec for the runtime image. We include `marimo` for notebook
# functionality and `uv` for fast package management.

import os
from flytekit import Resources, ImageSpec

from union.app import App

img = ImageSpec(
    name="marimo",
    builder="unionai",
    packages=[
        "marimo",
        "union-runtime",
        "union",
        "uv",
    ],
)

# We define the application configuration to serve static WASM files. The app uses Python's
# built-in HTTP server to serve the exported WASM files from the `wasm/` directory.
# The `include` parameter ensures all WASM files are packaged with the application.

marimo = App(
    name="marimo-wasm",
    container_image=img,
    args=[
        "python",
        "-m",
        "http.server",
        "--directory",
        "wasm/",
        "3003" #port
    ],
    include=["wasm/**"],
    port=3003,
    limits=Resources(cpu="1", mem="2Gi"),
    env={
        "UV_CACHE_DIR": "/root",
    },
    min_replicas=1,
)

# ## Step-by-Step Instructions
#
# ### Step 1: Write a Marimo Notebook
#
# Create a Marimo notebook similar to `marimo_notebook.py`. This example includes:
# - Interactive plots using matplotlib
# - Reactive cells that update when dependencies change
# - Markdown documentation cells
#
# ```python
# import marimo as mo
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Create interactive plots and visualizations
# x = np.linspace(0, 10)
# plt.plot(x, x**2)
# plt.gca()
# ```
#
# You can run and test your notebook locally with:
# ```shell
# $ marimo run marimo_notebook.py
# ```
#
# ### Step 2: Export as WASM Component
#
# Export your notebook as a WebAssembly application that can run in the browser:
#
# ```shell
# $ marimo export html-wasm marimo_notebook.py --output wasm/
# ```
#
# This command:
# - Converts your notebook to a standalone WASM application
# - Generates all necessary files in the `wasm/` directory
# - Creates an `output.html` file that serves as the entry point
#
# ### Step 3: Deploy the Application
#
# Deploy the application to Union Serving:
#
# ```shell
# $ union deploy apps app.py marimo-wasm
# ```
#
# The deployment will stream the status and provide you with an endpoint URL:
#
# ```console
# ✨ Deploying Application: marimo-wasm
# 🔎 Console URL: https://<union-tenant>/console/projects/<project>/domains/development/apps/marimo-wasm
# [Status] Started: Service is ready
# 🚀 Deployed Endpoint: https://your-app-endpoint.apps.<union-tenant>
# ```
#
# ### Step 4: Access Your Interactive Notebook
#
# Once deployed, access your interactive notebook by navigating to:
#
# ```
# https://your-app-endpoint.apps.<union-tenant>/output.html
# ```
#
# The notebook will load and run entirely in your browser, providing full interactivity
# without requiring a server-side Python runtime.
#
# ## Benefits of WASM Deployment
#
# - **No Server Dependencies**: Runs entirely in the browser
# - **Fast Loading**: No server round-trips for computations
# - **Scalable**: Can handle many concurrent users without server load
# - **Shareable**: Easy to distribute interactive notebooks
# - **Offline Capable**: Can work without internet connectivity once loaded
#
# ## Example Notebook Features
#
# The included `marimo_notebook.py` demonstrates:
# - Creating reactive plots with matplotlib
# - Using markdown cells for documentation
# - Implementing interactive data visualizations
# - Proper dependency management between cells
