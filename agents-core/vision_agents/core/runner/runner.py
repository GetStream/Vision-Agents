import asyncio
import logging
import warnings
from typing import Optional
from uuid import uuid4

import click
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vision_agents.core import AgentLauncher
from vision_agents.core.utils.logging import (
    configure_fastapi_loggers,
    configure_sdk_logger,
)

from .http.api import lifespan, router
from .http.dependencies import (
    can_close_session,
    can_start_session,
    can_view_metrics,
    can_view_session,
    get_current_user,
)
from .http.options import ServeOptions

logger = logging.getLogger(__name__)

# TODO:
#   7. Figure out how to serialize the agent config into some dict
#   8. Docs
#   10. Fix  WARNING  | AudioQueue buffer limit exceeded spamming the logs

asyncio_logger = logging.getLogger("asyncio")


class Runner:
    """
    A class to run Agents.

    Use `.run()` to run a single agent as a console app.
    Use `.serve()` to start a basic HTTP server that spawns agents to calls.
    Use `.cli()` for the CLI interface


    Examples:
        ```python
        # agent.py
        launcher = AgentLauncher(...)

        if __name__ == "__main__":
            launcher.cli()

        # `python agent.py serve` will start an HTTP server
        # `python agent.py run` with run a single agent as a console app
        ```
    """

    def __init__(
        self,
        launcher: AgentLauncher,
        serve_options: Optional[ServeOptions] = None,
    ):
        """
        Init the Runner object.

        Args:
            launcher: instance of `AgentLauncher`
            serve_options: instance of `ServeOptions` to configure behavior in `serve` mode.
            run_options: instance of `RunOptions` to configure behavior in `run` mode.
        """
        self._launcher = launcher
        self._serve_options = serve_options or ServeOptions()

        if self._serve_options.fast_api:
            # If `fast_api` is passed, assume it's a custom one and it as-is.
            logger.warning(
                "A custom `fast_api` object is detected, skipping configuration step"
            )
            self.fast_api = self._serve_options.fast_api
        else:
            # Otherwise, initialize FastAPI ourselves
            self.fast_api = self._create_fastapi_app(options=self._serve_options)

    def run(
        self,
        call_type: str = "default",
        call_id: Optional[str] = None,
        debug: bool = False,
        log_level: str = "INFO",
        no_demo: bool = False,
        video_track_override: Optional[str] = None,
    ) -> None:
        """
        Run the agent as the console app with the specified configuration.
        Args:
            call_type: Call type for the video call
            call_id: Call ID for the video call (auto-generated if not provided)
            debug: Enable debug mode
            log_level: Set the logging level
            no_demo: Disable opening the demo UI
            video_track_override: Optional local video track override for debugging.
                This track will play instead of any incoming video track.

        Returns:
            None
        """
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        configure_sdk_logger(level=numeric_level)

        # Suppress dataclasses_json missing value RuntimeWarnings.
        # They pollute the output and cannot be fixed by the users.
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="dataclasses_json.core"
        )

        # Generate call ID if not provided
        if call_id is None:
            call_id = str(uuid4())

        async def _run():
            logger.info("🚀 Launching agent...")

            try:
                # Start the agent launcher.
                await self._launcher.start()

                # Create the agent
                agent = await self._launcher.launch()

                logger.info("✅ Agent warmed up and ready")

                # Open demo UI by default
                if (
                    not no_demo
                    and hasattr(agent, "edge")
                    and hasattr(agent.edge, "open_demo_for_agent")
                ):
                    logger.info("🌐 Opening demo UI...")
                    await agent.edge.open_demo_for_agent(agent, call_type, call_id)

                # Join call if join_call function is provided
                logger.info(f"📞 Joining call: {call_type}/{call_id}")
                session = await self._launcher.start_session(
                    call_id, call_type, video_track_override_path=video_track_override
                )
                await session.wait()
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, shutting down gracefully...")
            except Exception as e:
                logger.error(f"❌ Error running agent: {e}", exc_info=True)
                raise
            finally:
                await self._launcher.stop()

        asyncio_logger_level = asyncio_logger.level

        try:
            asyncio.run(_run(), debug=debug)
        except KeyboardInterrupt:
            # Temporarily suppress asyncio error logging during cleanup
            asyncio_logger_level = asyncio_logger.level
            # Suppress KeyboardInterrupt and asyncio errors during cleanup
            asyncio_logger.setLevel(logging.CRITICAL)
            logger.info("👋 Agent shutdown complete")
        finally:
            # Restore original logging level
            asyncio_logger.setLevel(asyncio_logger_level)

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        agents_log_level: str = "INFO",
        http_log_level: str = "INFO",
    ):
        """
        Start the HTTP server that spawns agents to the calls.

        Args:
            host:
            port:
            agents_log_level:
            http_log_level:

        Returns:

        """
        # Configure loggers if they're not already configured
        configure_sdk_logger(
            level=getattr(logging, agents_log_level.upper(), logging.INFO)
        )
        configure_fastapi_loggers(
            level=getattr(logging, http_log_level.upper(), logging.INFO)
        )

        # Suppress dataclasses_json missing value RuntimeWarnings.
        # They pollute the output and cannot be fixed by the users.
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="dataclasses_json.core"
        )
        uvicorn.run(self.fast_api, host=host, port=port, log_config=None)

    def _create_fastapi_app(self, options: ServeOptions) -> FastAPI:
        app = FastAPI(lifespan=lifespan)
        app.state.launcher = self._launcher
        app.state.options = self._serve_options

        # Use dependency_overrides to allow passing free-form dependency functions
        # via ServeOptions.
        # This way, individual permission callables can define their own dependencies making them very flexible.
        app.dependency_overrides[can_start_session] = options.can_start_session
        app.dependency_overrides[can_close_session] = options.can_close_session
        app.dependency_overrides[can_view_session] = options.can_view_session
        app.dependency_overrides[can_view_metrics] = options.can_view_metrics
        app.dependency_overrides[get_current_user] = options.get_current_user
        app.include_router(router)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(options.cors_allow_origins),
            allow_credentials=options.cors_allow_credentials,
            allow_methods=list(options.cors_allow_methods),
            allow_headers=list(options.cors_allow_headers),
        )
        return app

    def cli(self):
        """
        Run the CLI
        """

        @click.group()
        @click.pass_context
        def cli_(ctx): ...

        @cli_.command()
        @click.option(
            "--call-type",
            type=str,
            default="default",
            help="Call type for the video call",
        )
        @click.option(
            "--call-id",
            type=str,
            default=None,
            help="Call ID for the video call (auto-generated if not provided)",
        )
        @click.option(
            "--debug",
            is_flag=True,
            default=False,
            help="Enable debug mode",
        )
        @click.option(
            "--log-level",
            type=click.Choice(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
            ),
            default="INFO",
            help="Set the logging level",
        )
        @click.option(
            "--no-demo",
            is_flag=True,
            default=False,
            help="Disable opening the demo UI",
        )
        @click.option(
            "--video-track-override",
            type=click.Path(dir_okay=False, exists=True, resolve_path=True),
            default=None,
            help="Optional local video track override for debugging. "
            "This track will play instead of any incoming video track.",
        )
        def run_cmd(
            call_type: str,
            call_id: Optional[str],
            debug: bool,
            log_level: str,
            no_demo: bool,
            video_track_override: Optional[str],
        ) -> None:
            """
            Run a single agent in the console.
            """
            return self.run(
                call_type=call_type,
                call_id=call_id,
                debug=debug,
                log_level=log_level,
                no_demo=no_demo,
                video_track_override=video_track_override,
            )

        @cli_.command()
        @click.option(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Server host",
        )
        @click.option(
            "--port",
            type=int,
            default=8000,
            help="Server port",
        )
        @click.option(
            "--agents-log-level",
            type=click.Choice(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
            ),
            default="INFO",
            help="Set the agents logging level",
        )
        @click.option(
            "--http-log-level",
            type=click.Choice(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
            ),
            default="INFO",
            help="Set the logging level for FastAPI and uvicorn",
        )
        def serve_cmd(
            host: str,
            port: int,
            agents_log_level: str,
            http_log_level: str,
        ) -> None:
            """
            Start the HTTP server that spawns agents to the calls.
            """
            return self.serve(
                host=host,
                port=port,
                agents_log_level=agents_log_level,
                http_log_level=http_log_level,
            )

        cli_()
