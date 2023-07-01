import functools
from typing import Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Extra, ValidationError
from sanic.exceptions import SanicException
from sanic.log import logger
from sanic.request import Request


class Params(BaseModel, extra=Extra.forbid):
    user: Union[str, int]
    n_rec: int
    user_feats: Optional[Dict[str, Union[str, int, float]]] = None
    seq: Optional[List[Union[str, int]]] = None


def validate(model: Type[object]):
    def decorator(func: Callable):
        @functools.wraps(func)
        async def decorated_function(request: Request, **kwargs):
            try:
                params = model(**request.json)
                kwargs["params"] = params
            except ValidationError as e:
                logger.error(f"Invalid request body: {request.json}")
                raise SanicException(
                    f"Invalid payload: `{request.json}`, please check key name and value type."
                ) from e

            return await func(request, **kwargs)

        return decorated_function

    return decorator
