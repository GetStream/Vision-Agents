import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vision_agents.core import AgentLauncher
from vision_agents.core.utils.logging import configure_sdk_logger

from .routes import router

logger = logging.getLogger(__name__)

# TODO:
#   1. HTTP server also needs a CLI, so we should extract it
#   2. CORS
#   3. Docs URL
#   4. Make sure users can setup their Auth

# TODO: Maybe if the fastapi is provided, we don't configure it at all?


class HTTPServerRunner:
    def __init__(
        self,
        launcher: AgentLauncher,
        fast_api: Optional[FastAPI] = None,
    ):
        self._launcher = launcher
        self._fast_api = fast_api or FastAPI()

    async def worker(self): ...

    def run(self, host: str = "127.0.0.1", port: int = 8000, log_level: str = "INFO"):
        app = self._configure_fastapi_app()
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        configure_sdk_logger(level=numeric_level)
        # TODO: Configure logs to follow the same format

        uvicorn.run(app, host=host, port=port)

    def _configure_fastapi_app(self) -> FastAPI:
        app = self._fast_api
        app.state.launcher = self._launcher
        app.include_router(router)
        # TODO: Allow to control origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return self._fast_api
