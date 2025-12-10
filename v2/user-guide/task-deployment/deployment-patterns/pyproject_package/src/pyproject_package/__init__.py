# unionai-examples/v2/user-guide/task-deployment/deployment-patterns/pyproject_package/src/pyproject_package/__init__.py

"""PyProject Package - Example Python package with Flyte tasks.

This package demonstrates a realistic Python project structure with:
- Modular business logic separate from Flyte orchestration
- Async Flyte tasks
- External dependencies (httpx, numpy, pydantic)
- Entrypoint script pattern

Main modules:
- data.loader: Data loading from APIs and files
- data.processor: Data cleaning, transformation, and aggregation
- models.analyzer: Statistical analysis and reporting
- main: Flyte tasks and pipeline orchestration
"""

from pyproject_package.main import main

__version__ = "0.1.0"
__all__ = ["main"]
