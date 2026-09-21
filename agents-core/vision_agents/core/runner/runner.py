import asyncio
import logging
import os
import sys
import warnings
from typing import Optional
from uuid import uuid4

import click
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vision_agents.core import AgentLauncher
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
from .simulate import (
    DEFAULT_JUDGE,
    DEFAULT_MAX_TURNS,
    DEFAULT_REPORT_DIR,
    run_simulation,
)

logger = logging.getLogger(__name__)

asyncio_logger = logging.getLogger("asyncio")

_SPLASH = """\
░█░█░▀█▀░█▀▀░▀█▀░█▀█░█▀█░░░█▀█░█▀▀░█▀▀░█▀█░▀█▀░█▀▀
░▀▄▀░░█░░▀▀█░░█░░█░█░█░█░░░█▀█░█░█░█▀▀░█░█░░█░░▀▀█
░░▀░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░░░▀░▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀▀▀"""


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
    Use `.simulate()` to run scenario files against the agent in text mode.
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
        # `python agent.py simulate scenarios/` runs YAML scenarios and writes a report
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

                logger.info("✅ Agent warmed up and ready")

                # Join call if join_call function is provided
                logger.info(f"📞 Joining call: {call_type}/{call_id}")
                session = await self._launcher.start_session(
                    call_id, call_type, video_track_override_path=video_track_override
                )
                # Open demo UI by default
                agent = session.agent
                if (
                    not no_demo
                    and hasattr(agent, "edge")
                    and hasattr(agent.edge, "open_demo_for_agent")
                ):
                    logger.info("🌐 Opening demo UI...")
                    await agent.edge.open_demo_for_agent(agent, call_type, call_id)

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

    def simulate(
        self,
        target: str,
        repeat: Optional[int] = None,
        variations: Optional[int] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        judge: str = DEFAULT_JUDGE,
        report_dir: str = DEFAULT_REPORT_DIR,
        name_filter: Optional[str] = None,
        log_level: str = "WARNING",
    ) -> int:
        """
        Run scenario files against the agent in text mode and write a report.

        A simulated user plays every scenario against a fresh agent, a judge
        model rules on each success criterion, and ``report.json`` /
        ``report.md`` land in ``report_dir`` with the full transcripts and verdicts.

        Args:
            target: A scenario file or a directory of ``*.yaml`` scenario files.
            repeat: How many times to run every variation; overrides the scenario file.
            variations: Ways of phrasing each scenario; overrides the scenario file.
            max_turns: Maximum number of simulated-user messages per conversation.
            judge: Model that plays the user and judges: ``provider/model`` or ``module:attribute``.
            report_dir: Directory for ``report.json`` and ``report.md``.
            name_filter: Only run scenarios whose name contains this text.
            log_level: Logging level while the simulation runs.

        Returns:
            The exit code: 0 when every scenario passed, 1 when any failed, 2 when
            a case never reached a verdict.

        Raises:
            click.ClickException: if the scenarios, ``judge`` or ``report_dir``
                cannot be used; its exit code is 2.
        """
        return run_simulation(
            self._launcher.launch,
            target,
            repeat=repeat,
            variations=variations,
            max_turns=max_turns,
            judge=judge,
            report_dir=report_dir,
            name_filter=name_filter,
            log_level=log_level,
        )

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

        @cli_.command()
        @click.argument(
            "target",
            metavar="SCENARIOS",
            type=click.Path(exists=True, resolve_path=True),
        )
        @click.option(
            "--repeat",
            type=click.IntRange(min=1),
            default=None,
            help="Run every variation this many times. Overrides the scenario file.",
        )
        @click.option(
            "--variations",
            type=click.IntRange(min=1),
            default=None,
            help="Ways of phrasing each scenario; the brief as written is always "
            "the first. Overrides the scenario file.",
        )
        @click.option(
            "--max-turns",
            type=click.IntRange(min=1),
            default=DEFAULT_MAX_TURNS,
            show_default=True,
            help="Maximum number of simulated-user messages per conversation.",
        )
        @click.option(
            "--judge",
            type=str,
            default=DEFAULT_JUDGE,
            show_default=True,
            metavar="MODEL",
            help="Model that plays the user and judges the transcripts: "
            "provider/model (e.g. gemini/gemini-2.5-flash) or module:attribute "
            "naming a callable that returns an LLM.",
        )
        @click.option(
            "--report",
            "report_dir",
            type=click.Path(file_okay=False),
            default=DEFAULT_REPORT_DIR,
            show_default=True,
            help="Directory for report.json and report.md.",
        )
        @click.option(
            "--filter",
            "name_filter",
            type=str,
            default=None,
            metavar="NAME",
            help="Only run scenarios whose name contains NAME (case-insensitive).",
        )
        @click.option(
            "--log-level",
            type=click.Choice(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
            ),
            default="WARNING",
            help="Set the logging level",
        )
        def simulate_cmd(
            target: str,
            repeat: Optional[int],
            variations: Optional[int],
            max_turns: int,
            judge: str,
            report_dir: str,
            name_filter: Optional[str],
            log_level: str,
        ) -> None:
            """
            Run scenario files against the agent and report pass/fail.

            SCENARIOS is a scenario file or a directory of *.yaml scenario files.
            Exits 0 when every scenario passes, 1 when any fails and 2 when no
            verdict could be reached (judge or provider error, bad --judge,
            missing or malformed scenarios), so it can gate CI.
            """
            code = self.simulate(
                target,
                repeat=repeat,
                variations=variations,
                max_turns=max_turns,
                judge=judge,
                report_dir=report_dir,
                name_filter=name_filter,
                log_level=log_level.upper(),
            )
            if code:
                raise click.exceptions.Exit(code)

        cli_(args=args)
