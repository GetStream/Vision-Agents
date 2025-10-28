import asyncio
import base64
import json
import logging
import uuid
from typing import Optional, List, Dict, Any
from getstream.video.rtc.audio_track import AudioStreamTrack

from vision_agents.core.llm import realtime
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.processors import Processor
from vision_agents.core.edge.types import Participant
from vision_agents.core.edge.types import PcmData
from ...core.llm.events import RealtimeAudioOutputEvent

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "amazon.nova-sonic-v1:0"
DEFAULT_SAMPLE_RATE = 16000


"""
TODO:
- Implement & cleanup function calling
- Cleanup chat integration
"""


class Realtime(realtime.Realtime):
    """
    Realtime on AWS with support for audio/video streaming (uses AWS Bedrock).

    A few things are different about Nova compared to other STS solutions

        1. two init events. there is a session start and a prompt start
        2. promptName basically works like a unique identifier. it's created client side and sent to nova
        3. input/text events are wrapped. so its common to do start event, text event, stop event
        4. on close there is an session and a prompt end event

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
        
        # Connect to the session
        await llm.connect()
        
        # Simple text response
        await llm.simple_response("Describe what you see and say hi")
        
        # Send audio
        await llm.simple_audio_response(pcm_data)
        
        # Close when done
        await llm.close()
    """
    connected : bool = False
    voice_id : str

    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            region_name: str = "us-east-1",
            voice_id: str = "matthew",
            **kwargs
    ) -> None:
        """

        """
        super().__init__(**kwargs)
        self.model = model
        self.region_name = region_name
        self.sample_rate = 24000
        self.voice_id = voice_id

        # Initialize Bedrock Runtime client with SDK
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{region_name}.amazonaws.com",
            region=region_name,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)
        self.logger = logging.getLogger(__name__)

        # Audio output track - Bedrock typically outputs at 16kHz
        self.output_track = AudioStreamTrack(
            framerate=24000, stereo=False, format="s16"
        )

        self._video_forwarder: Optional[VideoForwarder] = None
        self._stream_task: Optional[asyncio.Task[Any]] = None
        self._is_connected = False
        self._message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._conversation_messages: List[Dict[str, Any]] = []
        self._pending_tool_uses: Dict[int, Dict[str, Any]] = {}  # Track tool calls across stream events

        # Audio streaming configuration
        self.prompt_name = self.session_id

    async def connect(self):
        """To connect we need to do a few things

        - start a bi directional stream
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

        # Initialize the stream
        logger.info("Connecting to AWS Bedrock for model %s", self.model)
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model)
        )
        self.connected = True

        # Start listener task
        self._stream_task = asyncio.create_task(self._handle_events())

        # send start and prompt event
        await self.start_session()
        await self.start_prompt()

        # next send system instructions
        system_instructions = self._build_enhanced_instructions()
        if not system_instructions:
            raise Exception("AWS Bedrock requires system instructions before sending regular user input")
        await self.content_input(system_instructions, "SYSTEM")

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ):
        """Send audio data to the model for processing."""
        if not self.connected:
            self.logger.warning("realtime is not active. can't call simple_audio_response")

        # Resample from 48kHz to 24kHz if needed
        pcm = pcm.resample(24000)
        
        content_name = str(uuid.uuid4())

        await self.audio_content_start(content_name)
        self._emit_audio_input_event(pcm.samples, sample_rate=pcm.sample_rate)
        # Convert PcmData to base64 encoded bytes
        audio_base64 = base64.b64encode(pcm.samples).decode('utf-8')
        await self.audio_input(content_name, audio_base64)

        await self.content_end(content_name)

    async def simple_response(self, text: str, processors: Optional[List[Processor]] = None,
                              participant: Optional[Participant] = None):
        """
        Simple response standardizes how to send a text instruction to this LLM.

        Example:
            llm.simple_response("tell me a poem about Boulder")

        For more advanced use cases you can use the native send_realtime_input
        """
        self.logger.info("Simple response called with text: %s", text)
        await self.content_input(content=text, role="USER")

    async def content_input(self, content: str, role: str):
        """
        For text input Nova expects content start, text input and then content end
        This method wraps the 3 events in one operation
        """
        content_name = str(uuid.uuid4())
        await self.text_content_start(content_name, role)
        await self.text_input(content_name, content)
        await self.content_end(content_name)

    async def audio_input(self, content_name: str, audio_bytes: str):
        audio_event = {
            "event": {
                "audioInput": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "content": audio_bytes
                }
            }
        }
        await self.send_event(audio_event)

    async def audio_content_start(self, content_name: str, role: str="USER"):
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
                    "encoding": "base64"
                }
            }
          }
        }
        await self.send_event(event)

    async def start_session(self):
        # subclass this to change the session start
        event = {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 10240, # TODO: make configurable in init?
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }

        await self.send_event(event)

    async def start_prompt(self):
        prompt_name = self.session_id
        event = {
          "event": {
            "promptStart": {
                "promptName": prompt_name,
                "textOutputConfiguration": {
                    "mediaType": "text/plain"
                },
                "audioOutputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": 24000,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "voiceId": self.voice_id,
                    "encoding": "base64",
                    "audioType": "SPEECH"
                }
            }
          }
        }
        # TODO: tool support
        await self.send_event(event)



    async def text_content_start(self, content_name: str, role: str):
        event = {
          "event": {
            "contentStart": {
                "promptName": self.session_id,
                "contentName": content_name,
                "type": "TEXT",
                "interactive": False,
                "role": role,
                "textInputConfiguration": {
                    "mediaType": "text/plain"
                }
            }
          }
        }
        await self.send_event(event)

    async def text_input(self, content_name: str, content: str):
        event = {
            "event": {
                "textInput": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                    "content": content,
                }
            }
        }
        await self.send_event(event)

    async def content_end(self, content_name: str):
        event = {
            "event": {
                "contentEnd": {
                    "promptName": self.session_id,
                    "contentName": content_name,
                }
            }
        }
        await self.send_event(event)

    async def send_event(self, event_data: Dict[str, Any]) -> None:
        event_json = json.dumps(event_data)
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)

    async def close(self):
        if not self.connected:
            return

        prompt_end = {
            "event": {
                "promptEnd": {
                    "promptName": self.session_id,
                }
            }
        }
        await self.send_event(prompt_end)

        session_end: Dict[str, Any] = {
            "event": {
                "sessionEnd": {}
            }
        }
        await self.send_event(session_end)

        await self.stream.input_stream.close()

        if self._stream_task:
            self._stream_task.cancel()

        self.connected = False


    async def _handle_events(self):
        """Process incoming responses from AWS Bedrock."""
        try:
            while True:
                try:
                    output = await self.stream.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            json_data = json.loads(response_data)

                            # Handle different response types
                            if 'event' in json_data:
                                if 'contentStart' in json_data['event']:
                                    content_start = json_data['event']['contentStart']
                                    logger.info(f"Content start from AWS Bedrock: {content_start}")
                                    # set role
                                    self.role = content_start['role']
                                    # Check for speculative content
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                            if additional_fields.get('generationStage') == 'SPECULATIVE':
                                                self.display_assistant_text = True
                                            else:
                                                self.display_assistant_text = False
                                        except json.JSONDecodeError:
                                            pass
                                elif 'textOutput' in json_data['event']:
                                    text_content = json_data['event']['textOutput']['content']
                                    #role = json_data['event']['textOutput']['role']
                                    logger.info(f"Text output from AWS Bedrock: {text_content}")
                                elif 'completionStart' in json_data['event']:
                                    logger.info("Completion start from AWS Bedrock", json_data['event']['completionStart'])
                                elif 'audioOutput' in json_data['event']:
                                    audio_content = json_data['event']['audioOutput']['content']
                                    audio_bytes = base64.b64decode(audio_content)
                                    #await self.audio_output_queue.put(audio_bytes)

                                    audio_event = RealtimeAudioOutputEvent(
                                        plugin_name="aws",
                                        audio_data=audio_bytes,
                                        sample_rate=24000
                                    )
                                    self.events.send(audio_event)

                                    await self.output_track.write(audio_bytes)


                                elif 'toolUse' in json_data['event']:
                                    logger.info(f"Tool use from AWS Bedrock: {json_data['event']['toolUse']}")
                                    self.toolUseContent = json_data['event']['toolUse']
                                    self.toolName = json_data['event']['toolUse']['toolName']
                                    self.toolUseId = json_data['event']['toolUse']['toolUseId']
                                    # Add tool use to chat history
                                    # TODO
                                    #self.chat_history.add_tool_call(
                                    #    tool_use_content=self.toolUseContent
                                    #)
                                elif 'contentEnd' in json_data['event'] and json_data['event'].get('contentEnd',
                                                                                                   {}).get(
                                        'type') == 'TOOL':
                                    logger.info(f"Tool end from AWS Bedrock: {json_data['event']['contentEnd']}")
                                    # TODO: Implement tool support
                                    # toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                                    # self.chat_history.add_tool_result(self.toolUseId, toolResult)
                                    # toolContent = str(uuid.uuid4())
                                    # await self.send_tool_start_event(toolContent)
                                    # await self.send_tool_result_event(toolContent, toolResult)
                                    # await self.send_tool_content_end_event(toolContent)
                                elif 'contentEnd' in json_data['event']:


                                    stopReason = json_data['event']['contentEnd']['stopReason']
                                    if stopReason == "INTERRUPTED":
                                        logger.info("TODO: should flush audio buffer")
                                    logger.info(f"Content end from AWS Bedrock {stopReason}: {json_data['event']['contentEnd']}")

                                elif 'completionEnd' in json_data['event']:
                                    logger.info(f"Completion end from AWS Bedrock: {json_data['event']['completionEnd']}")
                                    # Handle end of conversation, no more response will be generated
                                elif "usageEvent" in json_data['event']:
                                    #logger.info(f"Usage event from AWS Bedrock: {json_data['event']['usageEvent']}")
                                    pass
                                else:
                                    logger.warning(f"Unhandled event: {json_data['event']}")

                            # Put the response in the output queue for other components
                            #await self.output_queue.put(json_data)
                        except json.JSONDecodeError:
                            pass
                            #await self.output_queue.put({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    logger.error("Stop async iteration exception")
                    break
                except Exception as e:
                    logger.error("Error, %s", e)
                    # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        print(f"Validation error: {error_message}")
                    else:
                        print(f"Error receiving response: {e}")
                    break

        except Exception as e:
            logger.error("Error, %s", e)
            print(f"Response processing error: {e}")
        finally:
            self.connected = False

