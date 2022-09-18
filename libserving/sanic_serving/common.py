import functools
from typing import Callable, Type

from pydantic import BaseModel, ValidationError
from sanic.log import logger
from sanic.request import Request


class Params(BaseModel):
    user: str
    n_rec: int


def validate(model: Type[object]):
    def decorator(func: Callable):
        @functools.wraps(func)
        async def decorated_function(request: Request, **kwargs):
            try:
                params = model(**request.json)
                kwargs["params"] = params
            except ValidationError:
                logger.error(f"Invalid request body: {request.json}")
                raise
            return await func(request, **kwargs)

        return decorated_function

    return decorator
