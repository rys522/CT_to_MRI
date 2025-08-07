import inspect
import time
import traceback
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from .logger import logger

F = TypeVar("F", bound=Callable[..., Any])


def log_and_time() -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            logger.info(f"Starting {func.__name__} function")

            result: Any = func(*args, **kwargs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(
                f"Finished {func.__name__} function in {elapsed_time:.4f} seconds"
            )

            return result

        return wrapper

    return decorator


def error_wrap(func: F) -> F:
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as err:
            logger.error(  # noqa: TRY400
                f"Error in {func.__name__}: {err}\n{traceback.format_exc()}"
            )
            return None

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.error(  # noqa: TRY400
                f"Error in {func.__name__}: {err}\n{traceback.format_exc()}"
            )
            return None

    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    else:
        return cast(F, sync_wrapper)


class SingletonMeta(type):
    _instances = {}  # noqa: RUF012

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
