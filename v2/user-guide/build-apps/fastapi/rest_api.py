# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""Example REST API with multiple endpoints."""

from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import flyte
from flyte.app.extras import FastAPIAppEnvironment


# {{docs-fragment rest-api}}
app = FastAPI(title="Product API")


# Data models
class Product(BaseModel):
    id: int
    name: str
    price: float


class ProductCreate(BaseModel):
    name: str
    price: float


# In-memory database (use real database in production)
products_db = []


@app.get("/products", response_model=List[Product])
async def get_products():
    return products_db


@app.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: int):
    product = next((p for p in products_db if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.post("/products", response_model=Product)
async def create_product(product: ProductCreate):
    new_product = {
        "id": len(products_db) + 1,
        "name": product.name,
        "price": product.price,
    }
    products_db.append(new_product)
    return new_product


env = FastAPIAppEnvironment(
    name="product-api",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)
# {{/docs-fragment rest-api}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"API URL: {app_deployment[0].url}")
    print(f"Swagger docs: {app_deployment[0].url}/docs")
