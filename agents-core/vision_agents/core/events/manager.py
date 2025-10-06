import asyncio
import collections
import types
import typing
import logging
from typing import get_origin, Union, get_args, Deque, Dict, Any, Optional
from .base import ExceptionEvent, HealthCheckEvent, ConnectionOkEvent, ConnectionErrorEvent, ConnectionClosedEvent


logger = logging.getLogger(__name__)


def _truncate_event_for_logging(event, max_length=200):
    """
    Truncate event data for logging to prevent log spam.
    
    Args:
        event: The event object to truncate
        max_length: Maximum length of the string representation
        
    Returns:
        Truncated string representation of the event
    """
    event_str = str(event)
    
    # Special handling for audio data arrays
    if hasattr(event, 'pcm_data') and hasattr(event.pcm_data, 'samples'):
        # Replace the full array with a summary
        samples = event.pcm_data.samples
        array_summary = f"array([{samples[0]}, {samples[1]}, ..., {samples[-1]}], dtype={samples.dtype}, size={len(samples)})"
        event_str = event_str.replace(str(samples), array_summary)
    
    # If the event is still too long, truncate it
    if len(event_str) > max_length:
        # Find a good truncation point (end of a field)
        truncate_at = max_length - 20  # Leave room for "... (truncated)"
        while truncate_at > 0 and event_str[truncate_at] not in [',', ')', '}']:
            truncate_at -= 1
        
        if truncate_at > 0:
            event_str = event_str[:truncate_at] + "... (truncated)"
        else:
            event_str = event_str[:max_length-20] + "... (truncated)"
    
    return event_str



class EventManager:
    """
    A comprehensive event management system for handling asynchronous event-driven communication.
    
    The EventManager provides a centralized way to register events, subscribe handlers,
    and process events asynchronously. It supports event queuing, error handling,
    and automatic exception event generation.
    
    Features:
    - Event registration and validation
    - Handler subscription with type hints
    - Asynchronous event processing
    - Error handling with automatic exception events
    - Support for Union types in handlers
    - Event queuing and batch processing
    
    Example:
        ```python
        from vision_agents.core.events.manager import EventManager
        from vision_agents.core.vad.events import VADSpeechStartEvent, VADSpeechEndEvent
        from vision_agents.core.stt.events import STTTranscriptEvent
        from vision_agents.core.tts.events import TTSAudioEvent
        
        # Create event manager
        manager = EventManager()
        
        # Register events
        manager.register(VADSpeechStartEvent)
        manager.register(VADSpeechEndEvent)
        manager.register(STTTranscriptEvent)
        manager.register(TTSAudioEvent)
        
        # Subscribe to VAD events
        @manager.subscribe
        async def handle_speech_start(event: VADSpeechStartEvent):
            print(f"Speech started with probability {event.speech_probability}")
        
        @manager.subscribe
        async def handle_speech_end(event: VADSpeechEndEvent):
            print(f"Speech ended after {event.total_speech_duration_ms}ms")
        
        # Subscribe to STT events
        @manager.subscribe
        async def handle_transcript(event: STTTranscriptEvent):
            print(f"Transcript: {event.text} (confidence: {event.confidence})")
        
        # Subscribe to multiple event types using Union
        @manager.subscribe
        async def handle_audio_events(event: VADSpeechStartEvent | VADSpeechEndEvent):
            print(f"VAD event: {event.type}")
        
        # Send events
        manager.send(VADSpeechStartEvent(
            plugin_name="silero",
            speech_probability=0.95,
            activation_threshold=0.5
        ))
        manager.send(STTTranscriptEvent(
            plugin_name="deepgram",
            text="Hello world",
            confidence=0.98
        ))
        
        # Before shutdown, ensure all events are processed
        await manager.shutdown()
        ```
    
    Args:
        ignore_unknown_events (bool): If True, unknown events are ignored rather than raising errors.
            Defaults to True.
    """
    
    def __init__(self, ignore_unknown_events: bool = True):
        """
        Initialize the EventManager.
        
        Args:
            ignore_unknown_events (bool): If True, unknown events are ignored rather than raising errors.
                Defaults to True.
        """
        self._queue: Deque[Any] = collections.deque([])
        self._events: Dict[str, type] = {}
        self._handlers: Dict[type, typing.List[typing.Callable]] = {}
        self._modules: Dict[str, typing.List[type]] = {}
        self._ignore_unknown_events = ignore_unknown_events
        self._processing_task: Optional[asyncio.Task[Any]] = None
        self._shutdown = False

        self.register(ExceptionEvent)
        self.register(HealthCheckEvent)
        self.register(ConnectionOkEvent)
        self.register(ConnectionErrorEvent)
        self.register(ConnectionClosedEvent)
        
        # Start background processing task
        self._start_processing_task()

    def register(self, event_class, ignore_not_compatible=False):
        """
        Register an event class for use with the event manager.
        
        Event classes must:
        - Have a name ending with 'Event'
        - Have a 'type' attribute (string)
        
        Example:
            ```python
            from vision_agents.core.vad.events import VADSpeechStartEvent
            from vision_agents.core.stt.events import STTTranscriptEvent
            
            manager = EventManager()
            manager.register(VADSpeechStartEvent)
            manager.register(STTTranscriptEvent)
            ```
        
        Args:
            event_class: The event class to register
            ignore_not_compatible (bool): If True, log warning instead of raising error
                for incompatible classes. Defaults to False.
        
        Raises:
            ValueError: If event_class doesn't meet requirements and ignore_not_compatible is False
        """
        if event_class.__name__.endswith('Event') and hasattr(event_class, 'type'):
            #if event_class.type in self._events:
            #    raise KeyError(f"{event_class.type} is already registered.")
            self._events[event_class.type] = event_class
            logger.info(f"Registered new event {event_class} - {event_class.type}")
        elif not ignore_not_compatible:
            raise ValueError(f"Provide valid class that ends on '*Event' and 'type' attribute: {event_class}")
        else:
            logger.warning(f"Provide valid class that ends on '*Event' and 'type' attribute: {event_class}")

    def merge(self, em: 'EventManager'):
        # Stop the processing task in the merged manager
        if em._processing_task and not em._processing_task.done():
            em._processing_task.cancel()
        
        # Merge all data from the other manager
        self._events.update(em._events)
        self._modules.update(em._modules)
        self._handlers.update(em._handlers)
        for event in em._queue:
            self._queue.append(event)

        # NOTE: we are merged into one manager and all children
        # reference main one
        em._events = self._events
        em._modules = self._modules
        em._handlers = self._handlers
        em._queue = self._queue
        em._processing_task = None  # Clear the stopped task reference

    def register_events_from_module(self, module, prefix='', ignore_not_compatible=True):
        """
        Register all event classes from a module.
        
        Automatically discovers and registers all classes in a module that:
        - Have names ending with 'Event'
        - Have a 'type' attribute (optionally matching the prefix)
        
        Example:
            ```python
            # Register all VAD events from the core module
            from vision_agents.core import vad
            manager.register_events_from_module(vad.events, prefix="plugin.vad")
            
            # Register all TTS events from the core module
            from vision_agents.core import tts
            manager.register_events_from_module(tts.events, prefix="plugin.tts")
            
            # Register all events from a plugin module
            from vision_agents.plugins.silero import events as silero_events
            manager.register_events_from_module(silero_events, prefix="plugin.silero")
            ```
        
        Args:
            module: The Python module to scan for event classes
            prefix (str): Optional prefix to filter event types. Only events with
                types starting with this prefix will be registered. Defaults to ''.
            ignore_not_compatible (bool): If True, log warning instead of raising error
                for incompatible classes. Defaults to True.
        """
        for name, class_ in module.__dict__.items():
            if name.endswith('Event') and (not prefix or getattr(class_, 'type', '').startswith(prefix)):
                self.register(class_, ignore_not_compatible=ignore_not_compatible)
                self._modules.setdefault(module.__name__, []).append(class_)

    def _generate_import_file(self):
        import_file = []
        for module_name, events in self._modules.items():
            import_file.append(f"from {module_name} import (")
            for event in events:
                import_file.append(f"    {event.__name__},")
            import_file.append(")")
        import_file.append("")
        import_file.append("__all__ = [")
        for module_name, events in self._modules.items():
            for event in events:
                import_file.append(f'    "{event.__name__}",')
        import_file.append("]")
        import_file.append("")
        return import_file

    def unsubscribe(self, function):
        """
        Unsubscribe a function from all event types.
        
        Removes the specified function from all event handler lists.
        This is useful for cleaning up handlers that are no longer needed.
        
        Example:
            ```python
            @manager.subscribe
            async def speech_handler(event: VADSpeechStartEvent):
                print("Speech started")
            
            # Later, unsubscribe the handler
            manager.unsubscribe(speech_handler)
            ```
        
        Args:
            function: The function to unsubscribe from all event types.
        """
        # NOTE: not the efficient but will delete proper pointer to fucntion
        for funcs in self._handlers.values():
            try:
                funcs.remove(function)
            except ValueError:
                pass

    def subscribe(self, function):
        """
        Subscribe a function to handle specific event types.
        
        The function must have type hints indicating which event types it handles.
        Supports both single event types and Union types for handling multiple event types.
        
        Example:
            ```python
            # Single event type
            @manager.subscribe
            async def handle_speech_start(event: VADSpeechStartEvent):
                print(f"Speech started with probability {event.speech_probability}")
            
            # Multiple event types using Union
            @manager.subscribe
            async def handle_audio_events(event: VADSpeechStartEvent | VADSpeechEndEvent):
                print(f"VAD event: {event.type}")
            ```
        
        Args:
            function: The async function to subscribe as an event handler.
                Must have type hints for event parameters.
        
        Returns:
            The decorated function (for use as decorator).
        
        Raises:
            RuntimeError: If handler has multiple separate event parameters (use Union instead)
            KeyError: If event type is not registered and ignore_unknown_events is False
        """
        subscribed = False
        is_union = False
        annotations = typing.get_type_hints(function)

        for name, event_class in annotations.items():
            origin = get_origin(event_class)
            events: typing.List[type] = []

            if origin is Union or isinstance(event_class, types.UnionType):
                events = list(get_args(event_class))
                is_union = True
            else:
                events = [event_class]

            for sub_event in events:
                event_type = getattr(sub_event, "type", None)

                if subscribed and not is_union:
                    raise RuntimeError("Multiple seperated events per handler are not supported, use Union instead")

                if event_type in self._events:
                    subscribed = True
                    self._handlers.setdefault(event_type, []).append(function)
                    module_name = getattr(function, '__module__', 'unknown')
                    logger.info(f"Handler {function.__name__} from {module_name} registered for event {event_type}")
                elif not self._ignore_unknown_events:
                    raise KeyError(f"Event {sub_event} - {event_type} is not registered.")
                else:
                    module_name = getattr(function, '__module__', 'unknown')
                    logger.info(f"Event {sub_event} - {event_type} is not registered – skipping handler {function.__name__} from {module_name}.")
        return function

    def _prepare_event(self, event):
        if isinstance(event, dict):
            event_type = event.get('type', '')
            try:
                event_class = self._events[event_type]
                event = event_class.from_dict(event, infer_missing=True)  # type: ignore[attr-defined]
            except Exception:
                logger.exception(f"Can't convert dict {event} to event class, skipping")
                return

        if event.type in self._events:
            #logger.info(f"Received event {_truncate_event_for_logging(event)}")
            return event
        elif self._ignore_unknown_events:
                logger.info(f"Event not registered {_truncate_event_for_logging(event)}")
        else:
            raise RuntimeError(f"Event not registered {event}")

    def send(self, *events):
        """
        Send one or more events for processing.
        
        Events are added to the queue and will be processed by the background
        processing task. If an event handler raises an exception, an ExceptionEvent 
        is automatically created and queued for processing.
        
        Example:
            ```python
            # Send single event
            manager.send(VADSpeechStartEvent(
                plugin_name="silero",
                speech_probability=0.95,
                activation_threshold=0.5
            ))
            
            # Send multiple events
            manager.send(
                VADSpeechStartEvent(plugin_name="silero", speech_probability=0.95),
                STTTranscriptEvent(plugin_name="deepgram", text="Hello world")
            )
            
            # Send event from dictionary
            manager.send({
                "type": "plugin.vad_speech_start",
                "plugin_name": "silero",
                "speech_probability": 0.95
            })
            ```
        
        Args:
            *events: One or more event objects or dictionaries to send.
                Events can be instances of registered event classes or dictionaries
                with a 'type' field that matches a registered event type.
        
        Raises:
            RuntimeError: If event type is not registered and ignore_unknown_events is False
        """
        for event in events:
            event = self._prepare_event(event)
            if event:
                #logger.info(f"🎯 EventManager.send: {event.__class__.__name__} - {event.type}")
                self._queue.append(event)
        
    async def wait(self, timeout: float = 10.0):
        """
        Wait for all queued events to be processed.
        
        This is useful in tests to ensure events are processed before assertions.
        
        Args:
            timeout: Maximum time to wait for processing to complete
        """
        start_time = asyncio.get_event_loop().time()
        while self._queue and (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.01)
        
    def _start_processing_task(self):
        """Start the background event processing task."""
        if self._processing_task and not self._processing_task.done():
            return  # Already running
            
        loop = asyncio.get_running_loop()
        self._processing_task = loop.create_task(self._process_events_loop())

    async def _process_events_loop(self):
        """
        Background task that continuously processes events from the queue.
        
        This task runs until shutdown is requested and processes all events
        in the queue. It's shielded from cancellation to ensure all events
        are processed before shutdown.
        """
        while True:
            if self._queue:
                event = self._queue.popleft()
                try:
                    await self._process_single_event(event)
                except asyncio.CancelledError:
                    await self._process_single_event(event)
                    logger.info("Event processing task was cancelled, processing remaining events")
            else:
                await asyncio.sleep(0.0001)

    async def _process_single_event(self, event):
        """Process a single event."""
        for handler in self._handlers.get(event.type, []):
            try:
                module_name = getattr(handler, '__module__', 'unknown')
                #logger.info(f"Called handler {handler.__name__} from {module_name} for event {event.__class__.__name__}")
                await handler(event)
            except Exception as exc:
                self._queue.appendleft(ExceptionEvent(exc, handler))  # type: ignore[arg-type]
                module_name = getattr(handler, '__module__', 'unknown')
                logger.exception(f"Error calling handler {handler.__name__} from {module_name} for event {event.__class__.__name__}")

