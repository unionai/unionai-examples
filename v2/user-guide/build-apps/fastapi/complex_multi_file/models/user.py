# {{docs-fragment user-model}}
from pydantic import BaseModel


class User(BaseModel):
    id: int
    name: str
# {{/docs-fragment user-model}}
