import inspect
import sys
from pathlib import Path
from typing import Any

from loguru import logger as logurulogger

LOGURU_LEVEL = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
FORMAT = "{time:YYYY-MM-DD HH:mm:ss} <level>[{level}]</level> {message}"
LOGURU_LV_LOWER = [x.lower() for x in LOGURU_LEVEL]


class SingletonMeta(type):
    _instances: dict[Any, Any] = {}  # noqa: RUF012

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class CustomLogger(
    metaclass=SingletonMeta,
):
    def __getattr__(
        self,
        name: Any,
    ) -> None:
        if name not in LOGURU_LV_LOWER:
            return getattr(logurulogger, name)

        def custon_handler(
            message: Any,
        ) -> None:
            if name == "error":
                caller_name = inspect.stack()[1].function
                getattr(logurulogger, name)(f"Error in func {caller_name}")

            if isinstance(message, str):
                processed_message = message.split("\n")
                for _item in processed_message:
                    getattr(logurulogger, name)(_item)
            else:
                getattr(logurulogger, name)(message)

        return custon_handler

    def trace(self, message: Any) -> None:
        self.__getattr__("trace")(message)

    def debug(self, message: Any) -> None:
        self.__getattr__("debug")(message)

    def info(self, message: Any) -> None:
        self.__getattr__("info")(message)

    def success(self, message: Any) -> None:
        self.__getattr__("success")(message)

    def warning(self, message: Any) -> None:
        self.__getattr__("warning")(message)

    def error(self, message: Any) -> None:
        self.__getattr__("error")(message)

    def critical(self, message: Any) -> None:
        self.__getattr__("critical")(message)


def logger_add_handler(
    _logger: CustomLogger,
    file: Path | None = None,
    level: str | None = None,
) -> None:
    _logger.remove()
    if level is None:
        level = "TRACE"
    _logger.add(
        sys.stdout,
        colorize=True,
        format=FORMAT,
        level=level,
    )
    if file is None:
        return
    _logger.add(
        file,
        colorize=True,
        format=FORMAT,
        level=level,
    )


logger = CustomLogger()
logger_add_handler(logger)
