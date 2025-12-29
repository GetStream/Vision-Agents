import asyncio
import base64
import datetime
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import aiortc
from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from getstream.video.rtc import PcmData
from getstream.video.rtc.audio_track import AudioStreamTrack
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
from vision_agents.core.agents.agent_types import AgentOptions
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import realtime
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.vad.silero import SileroVADSession, SileroVADSessionPool
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "amazon.nova-2-sonic-v1:0"
# Reconnect after 5 seconds if there is silence. If there is no break in speech reconnect after 7 seconds
FORCE_RECONNECT_IN_MINUTES = 7.0


class RealtimeConnection:
    """Encapsulates a single AWS Bedrock bidirectional stream connection.

    This class manages the lifecycle of a single connection, including sending
    events and receiving responses. It can be replaced to enable reconnection.
    """

    def __init__(self, client: BedrockRuntimeClient, model_id: str):
        self.client = client
        self.model_id = model_id
        self.stream = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Initialize the bidirectional stream."""
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )

    async def send_event(self, event_data: Dict[str, Any]) -> None:
        """Send an event to AWS Nova.

        Args:
            event_data: Dictionary containing the event data to send
        """
        if not self.stream:
            raise RuntimeError("Connection not initialized. Call connect() first.")

        event_json = json.dumps(event_data)
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream.input_stream.send(event)

    async def await_output(self):
        """Wait for and return output from the stream."""
        if not self.stream:
            raise RuntimeError("Connection not initialized. Call connect() first.")
        return await self.stream.await_output()

    async def close(self):
        """Close the stream connection."""
        if self.stream:
            try:
                await self.stream.input_stream.close()
            finally:
                self.stream = None


class Realtime(realtime.Realtime, Warmable[SileroVADSessionPool]):
    """
    Realtime on AWS with support for audio streaming and function calling (uses AWS Bedrock).

    A few things are different about Nova compared to other STS solutions

        1. two init events. there is a session start and a prompt start
        2. promptName basically works like a unique identifier. it's created client side and sent to nova
        3. input/text events are wrapped. so its common to do start event, text event, stop event
        4. on close there is an session and a prompt end event

    Function Calling:
        This implementation supports AWS Nova's tool use feature. Register functions using
        the @llm.register_function decorator and they will be automatically made available
        to the model. When the model calls a function, it will be executed and the result
        sent back to continue the conversation.

    AWS Nova samples are the best docs:

        simple: https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic_simple.py
        full: https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic.py
        tool use: https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic_tool_use.py

    Input event docs: https://docs.aws.amazon.com/nova/latest/userguide/input-events.html
    Available voices are documented here:
    https://docs.aws.amazon.com/nova/latest/userguide/available-voices.html

    Resumption example:
    https://github.com/aws-samples/amazon-nova-samples/tree/main/speech-to-speech/repeatable-patterns/resume-conversation



    Examples:

        from vision_agents.plugins import aws

        llm = aws.Realtime(
            model="us.amazon.nova-sonic-v1:0",
            region_name="us-east-1"
        )

        # Register a custom function
        @llm.register_function(
            name="get_weather",
            description="Get weather for a city"
        )
        def get_weather(city: str) -> dict:
            return {"city": city, "temp": 72, "condition": "sunny"}

        # Connect to the session
        await llm.connect()

        # Simple text response
        await llm.simple_response("Describe what you see and say hi")

        # Send audio
        await llm.simple_audio_response(pcm_data)

        # Close when done
        await llm.close()
    """

    voice_id: str
    options: AgentOptions

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        region_name: str = "us-east-1",
        voice_id: str = "matthew",
        reconnect_after_minutes=5.0,  # Attempt to reconnect during silence after 5 minutes. Reconnect is forced after 7 minutes
        **kwargs,
    ) -> None:
        """ """
        super().__init__(**kwargs)
        self.model = model
        self.region_name = region_name
        self.sample_rate = 24000
        self.voice_id = voice_id
        self.reconnect_after_minutes = reconnect_after_minutes
        self.connected: bool = False
        self.last_connected_at: Optional[datetime.datetime] = None
        # when we last hear or send audio (track this for reconnection logic)
        self._last_audio_at: Optional[datetime.datetime] = None
        self._vad_session: Optional[SileroVADSession] = None

        # Initialize Bedrock Runtime client with SDK
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{region_name}.amazonaws.com",
            region=region_name,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)
        self.logger = logging.getLogger(__name__)
        self.prompt_name = self.session_id

        # Audio output track - Bedrock typically outputs at 24kHz
        self.output_audio_track = AudioStreamTrack(
            sample_rate=24000, channels=1, format="s16"
        )

        # Connection management
        self.connection: Optional[RealtimeConnection] = None
        self._handle_event_task: Optional[asyncio.Task[Any]] = None
        self._reconnection_check_task: Optional[asyncio.Task[Any]] = None
        self._reconnecting: bool = False
        self._pending_tool_calls: Dict[
            str, Dict[str, Any]
        ] = {}  # Store tool calls until contentEnd: key=toolUseId

    async def on_warmup(self) -> SileroVADSessionPool:
        return await SileroVADSessionPool.load(self.options.model_dir)

    def on_warmed_up(self, vad_pool: SileroVADSessionPool) -> None:
        # Initialize a new VAD session using the shared Silero VAD pool
        self._vad_session = vad_pool.session()

    def _attach_agent(self, agent):
        logger.info(f"Attaching agent {agent}")
        super()._attach_agent(agent)
        self.options = agent.options

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        # No video support for now.
        return None

    async def connect(self):
        """To connect we need to do a few things

        - start a bidirectional stream
        - send session start event
        - send prompt start event
        - send text content start, text content, text content end

        Two unusual things here are that you have
        - 2 init events (session and prompt start)
        - text content is wrapped

        The init events should be easy to customize
        """
        if self.connected:
            self.logger.warning("Already connected")
            return

        # Initialize the connection
        logger.info("Connecting to AWS Bedrock for model %s", self.model)
        self.connection = RealtimeConnection(self.client, self.model)
        await self.connection.connect()

        self.last_connected_at = datetime.datetime.now()

        # Start listener task
        self._handle_event_task = asyncio.create_task(self._handle_events())

        # Start reconnection check task
        self._reconnection_check_task = asyncio.create_task(
            self._reconnection_check_loop()
        )

        # send start and prompt event
        event = self._create_session_start_event()
        await self.connection.send_event(event)

        # Small delay between init events
        await asyncio.sleep(0.1)

        await self.start_prompt()

        # Give AWS Nova a moment to process the prompt start event
        await asyncio.sleep(0.1)

        # next send system instructions
        if not self._instructions:
            raise ValueError(
                "AWS Bedrock requires system instructions before sending regular user input"
            )
        await self.content_input(self._instructions, "SYSTEM")

        self.connected = True
        logger.info("AWS Bedrock connection established")

    async def reconnect(self):
        """Reconnect to AWS Bedrock with a new connection.

        Creates a new connection, sets it up, then closes the old one.
        This allows seamless transition without dropping the conversation.
        """
        if self._reconnecting:
            logger.warning("Reconnection already in progress, skipping")
            return

        self._reconnecting = True
        reconnect_succeeded = False
        try:
            logger.info("Reconnecting to AWS Bedrock")

            # Create and initialize new connection
            new_connection = RealtimeConnection(self.client, self.model)
            await new_connection.connect()

            # Send session start and prompt start events on NEW connection before swapping
            session_start_event = self._create_session_start_event()
            await new_connection.send_event(session_start_event)
            prompt_start_event = self._create_prompt_start_event()
            await new_connection.send_event(prompt_start_event)

            # Now swap the connections
            old_connection = self.connection
            self.connection = new_connection
            # Update timestamp
            self.last_connected_at = datetime.datetime.now()

            # stop old events
            if self._handle_event_task:
                self._handle_event_task.cancel()
            # create the new one
            self._handle_event_task = asyncio.create_task(self._handle_events())

            # Resend system instructions
            if self._instructions:
                await self.content_input(self._instructions, "SYSTEM")

            # TODO: Resend summary of conversation history if needed

            # Close old connection
            if old_connection:
                await old_connection.close()

            reconnect_succeeded = True
            logger.info("Reconnection successful")
        finally:
            self._reconnecting = False
            if not reconnect_succeeded:
                self.connected = False

    def _should_reconnect(self) -> bool:
        """Check if connection should be reconnected based on runtime.

        AWS Nova has an 8 minute window limit. We should reconnect:
        - After 5 minutes if there's been silence (>3 seconds since last audio)
        - After 7 minutes regardless

        Returns:
            True if should reconnect, False otherwise
        """
        # Don't reconnect if already in progress
        if self._reconnecting:
            return False

        if not self.last_connected_at:
            return False

        now = datetime.datetime.now()
        running = now - self.last_connected_at
        running_minutes = running.total_seconds() / 60

        # Check if there's been silence (more than 3 seconds since last audio)
        has_silence = False
        if self._last_audio_at:
            silence_duration = (now - self._last_audio_at).total_seconds()
            has_silence = silence_duration > 3

        should_reconnect = False
        if running_minutes > self.reconnect_after_minutes and has_silence:
            should_reconnect = True
        elif running_minutes > FORCE_RECONNECT_IN_MINUTES:
            should_reconnect = True

        logger.info(
            "Connection is %.2f seconds old. Silence: %s, should reconnect is %s",
            running.total_seconds(),
            has_silence,
            should_reconnect,
        )

        return should_reconnect

    async def _reconnection_check_loop(self):
        """Periodic task that checks if reconnection is needed.

        Runs every second to check if the connection should be reconnected
        based on runtime and silence detection.
        """
        try:
            while self.connected:
                await asyncio.sleep(1)
                if self._should_reconnect() and not self._reconnecting:
                    logger.info("Reconnection needed, initiating reconnect...")
                    await self.reconnect()
        except asyncio.CancelledError:
            logger.debug("Reconnection check loop cancelled")

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ):
        """Send audio data to the model for processing."""
        if not self.connected:
            self.logger.warning(
                "realtime is not active. can't call simple_audio_response"
            )
            return

        if self._vad_session is None:
            raise ValueError("The VAD model has not been initialized.")

        # Resample to 24kHz if needed, as required by AWS Nova
        pcm = pcm.resample(24000)
        is_talking = self._vad_session.predict_speech(pcm) > 0.5
        if is_talking:
            self._last_audio_at = datetime.datetime.now()

        content_name = str(uuid.uuid4())

        await self.audio_content_start(content_name)
        self._emit_audio_input_event(pcm)

        # Convert PcmData to base64 encoded bytes
        audio_base64 = base64.b64encode(pcm.samples).decode("utf-8")
        await self.audio_input(content_name, audio_base64)

        await self.content_end(content_name)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ):
        """
        Simple response standardizes how to send a text instruction to this LLM.

        Example:
            llm.simple_response("tell me a poem about Boulder")

        For more advanced use cases you can use the native send_realtime_input
        """
        self.logger.info("Simple response called with text: %s", text)
        await self.content_input(content=text, role="USER")

    async def content_input(self, content: str, role: str, interactive: bool = True):
        """
        For text input Nova expects content start, text input and then content end
        This method wraps the 3 events in one operation
        """
        content_name = str(uuid.uuid4())
        logger.debug(f"Sending content input: role={role}, content={content[:100]}...")
        await self.text_content_start(content_name, role, interactive)
        await self.text_input(content_name, content)
        await self.content_end(content_name)

    async def audio_input(self, content_name: str, audio_bytes: str):
        if not self.connection:
            raise RuntimeError("Not connected")
        audio_event = {
            "event": {
                "audioInput": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "content": audio_bytes,
                }
            }
        }
        await self.connection.send_event(audio_event)

    async def audio_content_start(self, content_name: str, role: str = "USER"):
        if not self.connection:
            raise RuntimeError("Not connected")
        event = {
            "event": {
                "contentStart": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": role,
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        }
        await self.connection.send_event(event)

    def _create_session_start_event(self) -> Dict[str, Any]:
        """Create a session start event.

        Subclass this to customize the session start configuration.

        Returns:
            Event dictionary for session start
        """
        return {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    }
                },
                "turnDetectionConfiguration": {"endpointingSensitivity": "MEDIUM"},
            }
        }

    async def start_session(self):
        """Send a session start event."""
        if not self.connection:
            raise RuntimeError("Not connected")
        event = self._create_session_start_event()
        await self.connection.send_event(event)

    def _create_prompt_start_event(self) -> Dict[str, Any]:
        """Create a prompt start event.

        Returns:
            Event dictionary for prompt start
        """
        prompt_name = self.session_id

        # Add tool configuration if tools are available
        tools = self._convert_tools_to_provider_format(self.get_available_functions())

        # Build the base event structure
        event = {
            "event": {
                "promptStart": {
                    "promptName": prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": self.voice_id,
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        }

        # Add tool configuration if tools are available
        if tools:
            self.logger.info(f"Adding tool configuration with {len(tools)} tools")
            event["event"]["promptStart"]["toolUseOutputConfiguration"] = {
                "mediaType": "application/json"
            }
            event["event"]["promptStart"]["toolConfiguration"] = {"tools": tools}

        return event

    async def start_prompt(self, connection: Optional[RealtimeConnection] = None):
        """Send a prompt start event.

        Args:
            connection: Optional connection to send on. Uses self.connection if not provided.
        """
        event = self._create_prompt_start_event()
        target_connection = connection or self.connection
        if not target_connection:
            raise RuntimeError("Not connected")
        await target_connection.send_event(event)

    async def text_content_start(self, content_name: str, role: str, interactive: bool):
        if not self.connection:
            raise RuntimeError("Not connected")
        event = {
            "event": {
                "contentStart": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "type": "TEXT",
                    "role": role,
                    "interactive": interactive,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        }

        await self.connection.send_event(event)

    async def text_input(self, content_name: str, content: str):
        if not self.connection:
            raise RuntimeError("Not connected")
        event = {
            "event": {
                "textInput": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "content": content,
                }
            }
        }

        await self.connection.send_event(event)

    async def content_end(self, content_name: str):
        if not self.connection:
            raise RuntimeError("Not connected")
        event = {
            "event": {
                "contentEnd": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                }
            }
        }

        await self.connection.send_event(event)

    def _convert_tools_to_provider_format(
        self, tools: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert ToolSchema objects to AWS Nova Realtime format.

        Args:
            tools: List of ToolSchema objects from the function registry

        Returns:
            List of tools in AWS Nova Realtime format
        """
        aws_tools = []
        for tool in tools or []:
            name = tool.get("name", "unnamed_tool")
            description = tool.get("description", "") or ""
            params = tool.get("parameters_schema") or tool.get("parameters") or {}

            # Normalize to a valid JSON Schema object
            if not isinstance(params, dict):
                params = {}

            # Ensure it has the required JSON Schema structure
            if "type" not in params:
                # Extract required fields from properties if they exist
                properties = params if params else {}
                required = list(properties.keys()) if properties else []

                params = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            else:
                # Already has type, but ensure additionalProperties is set
                if "additionalProperties" not in params:
                    params["additionalProperties"] = False

            # AWS Nova expects toolSpec format with inputSchema.json as a JSON STRING (matching official example)
            # Convert the schema to a JSON string
            schema_json = json.dumps(params)

            aws_tool = {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {
                        "json": schema_json  # This should be a JSON string, not a dict
                    },
                }
            }
            aws_tools.append(aws_tool)
        return aws_tools

    async def send_tool_content_start(self, content_name: str, tool_use_id: str):
        """Send tool content start event.

        Args:
            content_name: Unique content identifier
            tool_use_id: The tool use ID from the toolUse event
        """
        if not self.connection:
            raise RuntimeError("Not connected")
        event = {
            "event": {
                "contentStart": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "type": "TOOL",
                    "interactive": False,
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        }
        await self.connection.send_event(event)

    async def send_tool_result(self, content_name: str, result: Any):
        """Send tool result event.

        Args:
            content_name: Unique content identifier
            result: The result from executing the tool (will be stringified as JSON)
        """
        if not self.connection:
            raise RuntimeError("Not connected")
        # AWS Nova expects content as a stringified JSON string
        # Reference: https://docs.aws.amazon.com/nova/latest/userguide/input-events.html
        if isinstance(result, str):
            content_str = result
        else:
            content_str = json.dumps(result)

        event = {
            "event": {
                "toolResult": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "content": content_str,  # Stringified JSON, not an object/array
                }
            }
        }
        await self.connection.send_event(event)

    async def _handle_tool_call(
        self, tool_name: str, tool_use_id: str, tool_use_content: Dict[str, Any]
    ):
        """Handle tool call from AWS Bedrock.

        Args:
            tool_name: Name of the tool to execute
            tool_use_id: The tool use ID from AWS
            tool_use_content: Full tool use content from AWS
        """
        logger.debug(f"Starting tool call execution: {tool_name} (id: {tool_use_id})")

        # Extract tool input from the tool use content (matching working example)
        tool_input = {}
        if "content" in tool_use_content:
            try:
                tool_input = json.loads(tool_use_content["content"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Could not parse tool input from content: {tool_use_content.get('content')}"
                )
                tool_input = {}
        elif "input" in tool_use_content:
            tool_input = tool_use_content["input"]

        # Create normalized tool call
        tool_call = {
            "type": "tool_call",
            "id": tool_use_id,
            "name": tool_name,
            "arguments_json": tool_input,
        }

        # Execute using existing tool execution infrastructure from base LLM
        tc, result, error = await self._run_one_tool(tool_call, timeout_s=30)

        # Prepare response data
        if error:
            response_data = {"error": str(error)}
            logger.error(f"Tool call {tool_name} failed: {error}")
        else:
            response_data = result

        # Send tool result back to AWS using Nova's format
        content_name = str(uuid.uuid4())

        await self.send_tool_content_start(content_name, tool_use_id)
        await self.send_tool_result(content_name, response_data)
        await self.content_end(content_name)

    async def close(self):
        if not self.connected:
            return

        if self.connection:
            prompt_end = {
                "event": {
                    "promptEnd": {
                        "promptName": self.session_id,
                    }
                }
            }
            await self.connection.send_event(prompt_end)

            session_end: Dict[str, Any] = {"event": {"sessionEnd": {}}}
            await self.connection.send_event(session_end)

            await self.connection.close()

        if self._handle_event_task:
            self._handle_event_task.cancel()

        if self._reconnection_check_task:
            self._reconnection_check_task.cancel()

        self.connected = False

    async def _handle_events(self):
        """Process incoming responses from AWS Bedrock."""
        while True:
            try:
                if not self.connection:
                    logger.warning("Connection lost, stopping event handler")
                    break
                output = await self.connection.await_output()
                result = await output[1].receive()
                if result.value and result.value.bytes_:
                    try:
                        response_data = result.value.bytes_.decode("utf-8")
                        json_data = json.loads(response_data)

                        # Handle different response types
                        if "event" in json_data:
                            # Log all non audio events
                            if (
                                "audioOutput" not in json_data["event"]
                                and "usageEvent" not in json_data["event"]
                            ):
                                logger.info(f"Received event: {json_data}")

                            if "contentStart" in json_data["event"]:
                                content_start = json_data["event"]["contentStart"]
                                logger.debug(
                                    f"Content start from AWS Bedrock: {content_start}"
                                )
                                # set role
                                self.role = content_start["role"]
                                # Check for speculative content
                                if "additionalModelFields" in content_start:
                                    try:
                                        additional_fields = json.loads(
                                            content_start["additionalModelFields"]
                                        )
                                        if (
                                            additional_fields.get("generationStage")
                                            == "SPECULATIVE"
                                        ):
                                            self.display_assistant_text = True
                                        else:
                                            self.display_assistant_text = False
                                    except json.JSONDecodeError:
                                        pass

                            elif "textOutput" in json_data["event"]:
                                text_content = json_data["event"]["textOutput"][
                                    "content"
                                ]
                                logger.debug(
                                    f"Text output from AWS Bedrock: {text_content}"
                                )
                            elif "completionStart" in json_data["event"]:
                                logger.debug(
                                    "Completion start from AWS Bedrock: %s",
                                    json_data["event"]["completionStart"],
                                )
                            elif "audioOutput" in json_data["event"]:
                                self._last_audio_at = datetime.datetime.now()
                                audio_content = json_data["event"]["audioOutput"][
                                    "content"
                                ]
                                audio_bytes = base64.b64decode(audio_content)
                                pcm = PcmData.from_bytes(audio_bytes, self.sample_rate)
                                self._emit_audio_output_event(
                                    audio_data=pcm,
                                )

                            elif "toolUse" in json_data["event"]:
                                tool_use_data = json_data["event"]["toolUse"]
                                tool_name = tool_use_data.get("toolName")
                                tool_use_id = tool_use_data.get("toolUseId")

                                logger.debug(
                                    f"Tool use event received: {tool_name} (id: {tool_use_id})"
                                )

                                # Store tool call info until contentEnd (matching working example)
                                if tool_use_id and tool_name:
                                    self._pending_tool_calls[tool_use_id] = {
                                        "toolName": tool_name,
                                        "toolUseId": tool_use_id,
                                        "toolUseContent": tool_use_data,
                                    }
                                else:
                                    logger.warning(
                                        f"Invalid tool use event - missing toolName or toolUseId: {tool_use_data}"
                                    )

                            elif "contentEnd" in json_data["event"]:
                                content_end_data = json_data["event"]["contentEnd"]
                                stop_reason = content_end_data.get("stopReason")
                                content_type = content_end_data.get("type")

                                logger.debug(
                                    f"Content end event: type={content_type}, stopReason={stop_reason}"
                                )

                                # Process tool calls on contentEnd with type == 'TOOL' (matching reference implementation)
                                if content_type == "TOOL":
                                    tool_use_id = content_end_data.get("toolUseId")

                                    # If toolUseId not in contentEnd, process most recent pending tool call
                                    if not tool_use_id and self._pending_tool_calls:
                                        # Get the most recently added tool call
                                        tool_use_id = list(
                                            self._pending_tool_calls.keys()
                                        )[-1]

                                    if (
                                        tool_use_id
                                        and tool_use_id in self._pending_tool_calls
                                    ):
                                        tool_call_info = self._pending_tool_calls.pop(
                                            tool_use_id
                                        )
                                        asyncio.create_task(
                                            self._handle_tool_call(
                                                tool_name=tool_call_info["toolName"],
                                                tool_use_id=tool_call_info["toolUseId"],
                                                tool_use_content=tool_call_info[
                                                    "toolUseContent"
                                                ],
                                            )
                                        )

                                if stop_reason == "INTERRUPTED":
                                    logger.debug("TODO: should flush audio buffer")
                                logger.debug(
                                    f"Content end from AWS Bedrock {stop_reason}: {content_end_data}"
                                )

                            elif "completionEnd" in json_data["event"]:
                                logger.debug(
                                    f"Completion end from AWS Bedrock: {json_data['event']['completionEnd']}"
                                )
                                # Handle end of conversation, no more response will be generated
                            elif "usageEvent" in json_data["event"]:
                                pass
                            else:
                                logger.warning(f"Unhandled event: {json_data['event']}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {e}")
            except StopAsyncIteration:
                # Stream has ended normally
                logger.debug("Stream ended normally")
                break
