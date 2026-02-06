# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "fastapi",
#    "pydantic",
# ]
# ///

"""Complete FastAPI app with authentication example."""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette import status
from pydantic import BaseModel
import os
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Authentication setup
# Get API key from environment variable (loaded from Flyte secret)
# The secret must be created using: flyte create secret api-key <your-api-key-value>
API_KEY = os.getenv("API_KEY")
security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> HTTPAuthorizationCredentials:
    """Verify Bearer token."""
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured",
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return credentials

# App setup
app = FastAPI(
    title="Authenticated Product API",
    description="API with authentication",
)

# Data models
class Product(BaseModel):
    id: int
    name: str
    price: float

class ProductCreate(BaseModel):
    name: str
    price: float

# In-memory database
products_db = []

# Public endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Protected endpoints
@app.get("/products", response_model=list[Product])
async def get_products(credentials = Depends(verify_token)):
    """Get all products (requires authentication)."""
    return products_db

@app.post("/products", response_model=Product)
async def create_product(
    product: ProductCreate,
    credentials = Depends(verify_token),
):
    """Create a new product (requires authentication)."""
    new_product = Product(
        id=len(products_db) + 1,
        name=product.name,
        price=product.price,
    )
    products_db.append(new_product)
    return new_product

# Environment configuration
env = FastAPIAppEnvironment(
    name="authenticated-product-api",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
        "pydantic",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,  # App-level authentication
    secrets=flyte.Secret(key="api-key", as_env_var="API_KEY"),
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed: {app_deployment[0].summary_repr()}")

