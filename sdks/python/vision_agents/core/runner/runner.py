import asyncio
import logging
import os
import sys
import warnings
import webbrowser
from typing import Optional
from uuid import uuid4

import click
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vision_agents.core import Agent, AgentLauncher
from vision_agents.core.llm import RemotePipeline
from vision_agents.core.utils import get_vision_agents_version
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
)
from .http.options import ServeOptions

logger = logging.getLogger(__name__)

asyncio_logger = logging.getLogger("asyncio")

_SPLASH = """\
░█░█░▀█▀░█▀▀░▀█▀░█▀█░█▀█░░░█▀█░█▀▀░█▀▀░█▀█░▀█▀░█▀▀
░▀▄▀░░█░░▀▀█░░█░░█░█░█░█░░░█▀█░█░█░█▀▀░█░█░░█░░▀▀█
░░▀░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░░░▀░▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀▀▀"""


# DASHBOARD_BASE_URL is where the local dashboard is served from. A call the router runs
# has a page there with the transcript, what the agent decided, and a button to join and
# talk to it.
_DASHBOARD_BASE_URL_ENV = "DASHBOARD_BASE_URL"
_DEFAULT_DASHBOARD_BASE_URL = "http://localhost:3000"

# How long to give a remote pipeline to join before opening the UI without it. Joining
# takes a few seconds, and the call is worth watching from the start.
_JOIN_WAIT_SECONDS = 20.0
_JOIN_POLL_SECONDS = 0.2


async def _open_ui(
    agent: Agent,
    call_type: str,
    call_id: str,
    join_task: Optional[asyncio.Task] = None,
) -> None:
    """Open whichever UI can show this call.

    A call running on the router has a dashboard page, keyed by the router's session
    rather than by the call id. A pipeline running in this process has no such page, so
    that falls back to the transport's own demo.
    """
    session_id = await _router_session_id(agent, join_task)
    if session_id:
        base = os.getenv(_DASHBOARD_BASE_URL_ENV, _DEFAULT_DASHBOARD_BASE_URL)
        url = f"{base.rstrip('/')}/calls/{session_id}"
        # Logged before opening, so the URL is still usable where there is no browser to
        # open it with, as in a container or over ssh.
        logger.info(f"🌐 Opening the dashboard: {url}")
        # A new tab, not the current window: a dashboard already open on an earlier call
        # would otherwise be raised without navigating, which looks like this call never
        # got a page.
        opened = await asyncio.to_thread(webbrowser.open, url, 2)
        if not opened:
            logger.warning("Could not open a browser; the call is at %s", url)
        return

    if hasattr(agent.edge, "open_demo_for_agent"):
        logger.info("🌐 Opening demo UI...")
        await agent.edge.open_demo_for_agent(agent, call_type, call_id)


async def _router_session_id(
    agent: Agent, join_task: Optional[asyncio.Task] = None
) -> Optional[str]:
    """Wait for the router to name the session it is running the call in.

    A remote pipeline joins in the background, so the session is not there the instant
    the call starts and there is nothing to wait on that does not also wait for a caller
    to arrive. Only a remote pipeline is waited for: a local one will never have a
    session, and holding the demo back for it would be a delay for nothing.
    """
    llm = agent.llm
    if llm is None or not isinstance(llm, RemotePipeline):
        return None

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _JOIN_WAIT_SECONDS
    while True:
        session_id = llm.router_session_id
        if session_id:
            return session_id
        if join_task is not None and join_task.done():
            logger.warning(
                "The pipeline did not join, so there is no dashboard page to open"
            )
            return None
        if loop.time() >= deadline:
            logger.warning(
                "The pipeline has not joined after %ss, so there is no dashboard page "
                "to open yet",
                _JOIN_WAIT_SECONDS,
            )
            return None
        await asyncio.sleep(_JOIN_POLL_SECONDS)


def _print_splash() -> None:
    """
    Print a splash screen.
    """
    banner_width = len(_SPLASH.splitlines()[0])
    click.echo()  # newline before the splash
    click.echo(click.style(_SPLASH, fg=(0, 95, 215), bold=True))
    # Align the version to the right side of the splash
    version = f"v{get_vision_agents_version()}".rjust(banner_width)
    click.echo(click.style(version, fg="white", dim=True))
    click.echo()


class Runner:
    """
    A class to run Agents.

    Use `.run()` to run a single agent as a console app.
    Use `.serve()` to start a basic HTTP server that spawns agents to calls.
    Use `.cli()` for the CLI interface


    Examples:
        ```python
        # agent.py
        from vision_agents.core import Runner, ServeOptions

        launcher = AgentLauncher(...)
        runner = Runner(launcher=launcher, serve_options=ServeOptions())

        if __name__ == "__main__":
            runner.cli()

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
        call_type: str = "agent",
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

                logger.info("✅ Agent warmed up and ready")

                # Join call if join_call function is provided
                logger.info(f"📞 Joining call: {call_type}/{call_id}")
                session = await self._launcher.start_session(
                    call_id, call_type, video_track_override_path=video_track_override
                )
                # Open demo UI by default
                agent = session.agent
                if not no_demo:
                    await _open_ui(agent, call_type, call_id, join_task=session.task)

                await session.wait()
            except asyncio.CancelledError:
                logger.info("The session is cancelled, shutting down gracefully...")
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
        debug: bool = False,
    ) -> None:
        """
        Start the HTTP server that spawns agents to the calls.

        Args:
            host: Host address to bind the server to.
            port: Port number for the server.
            agents_log_level: Logging level for agent-related logs.
            http_log_level: Logging level for FastAPI and uvicorn logs.
            debug: Enable asyncio debug mode.
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

        # Enable asyncio debug via environment variable before uvicorn creates its loop
        if debug:
            os.environ.setdefault("PYTHONASYNCIODEBUG", "1")
        uvicorn.run(self.fast_api, host=host, port=port, log_config=None)

    def _create_fastapi_app(self, options: ServeOptions) -> FastAPI:
        """
        Create and configure a FastAPI application for serving agents.

        Args:
            options: Configuration options for the server.

        Returns:
            Configured FastAPI application instance.
        """
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
        app.include_router(router)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(options.cors_allow_origins),
            allow_credentials=options.cors_allow_credentials,
            allow_methods=list(options.cors_allow_methods),
            allow_headers=list(options.cors_allow_headers),
        )
        return app

    def cli(self, args: Optional[list[str]] = None) -> None:
        """
        Run the command-line interface with `run` and `serve` subcommands.

        Args:
            args: Optional explicit argument list. When ``None``, Click reads
                ``sys.argv``. Pass a list (e.g. ``["run", "--debug"]``) to
                invoke the CLI programmatically without mutating ``sys.argv``.
        """

        @click.group()
        @click.pass_context
        def cli_(ctx):
            pass

        @cli_.command()
        @click.option(
            "--call-type",
            type=str,
            default="agent",
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
        @click.option(
            "--no-splash",
            is_flag=True,
            default=False,
            help="Disable the splash screen",
        )
        def run_cmd(
            call_type: str,
            call_id: Optional[str],
            debug: bool,
            log_level: str,
            no_demo: bool,
            video_track_override: Optional[str],
            no_splash: bool,
        ) -> None:
            """
            Run a single agent in the console.
            """
            if not no_splash and sys.stdout.isatty():
                _print_splash()
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
        @click.option(
            "--debug",
            is_flag=True,
            default=False,
            help="Enable asyncio debug mode",
        )
        @click.option(
            "--no-splash",
            is_flag=True,
            default=False,
            help="Disable the splash screen",
        )
        def serve_cmd(
            host: str,
            port: int,
            agents_log_level: str,
            http_log_level: str,
            debug: bool,
            no_splash: bool,
        ) -> None:
            """
            Start the HTTP server that spawns agents to the calls.
            """
            if not no_splash and sys.stdout.isatty():
                _print_splash()
            return self.serve(
                host=host,
                port=port,
                agents_log_level=agents_log_level.upper(),
                http_log_level=http_log_level.upper(),
                debug=debug,
            )

        cli_(args=args)
