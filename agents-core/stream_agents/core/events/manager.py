import collections
import dataclasses
import types
import typing
import logging
from typing import get_origin, Union, get_args



logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ExceptionEvent:
    exc: Exception
    handler: types.FunctionType
    type: str = 'event.exception'


class EventManager:
    def __init__(self, ignore_unknown_events: bool = True):
        self._queue = collections.deque([])
        self._events = {}
        self._handlers = {}
        self._modules = {}
        self._ignore_unknown_events = ignore_unknown_events

        self.register(ExceptionEvent)

    def register(self, event_class, ignore_not_compatible=False):
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
        self._events.update(em._events)
        self._modules.update(em._modules)
        self._handlers.update(em._handlers)
        for event in em._queue:
            self._queue.append(event)

        # NOTE: we are merged into one manager and all children
        # refrence main one
        em._events = self._events
        em._modules = self._modules
        em._handlers = self._handlers
        em._queue = self._queue

    def register_events_from_module(self, module, prefix='', ignore_not_compatible=True):
        for name, class_ in module.__dict__.items():
            if name.endswith('Event') and (not prefix or getattr(class_, 'type', '').startswith(prefix)):
                self.register(class_, ignore_not_compatible=ignore_not_compatible)
                self._modules.setdefault(module.__name__, []).append(class_)

    def _generate_import_file(self):
        import_file = []
        for module, events in self._modules.items():
            import_file.append(f"from {module.__name__} import (")
            for event in events:
                import_file.append(f"    {event.__name__},")
            import_file.append(")")
        import_file.append("")
        import_file.append("__all__ = [")
        for module, events in self._modules.items():
            for event in events:
                import_file.append(f'    "{event.__name__}",')
        import_file.append("]")
        import_file.append("")
        return import_file

    def unsubscribe(self, function):
        # NOTE: not the efficient but will delete proper pointer to fucntion
        for funcs in self._events.values():
            try:
                funcs.remove(function)
            except ValueError:
                pass

    def subscribe(self, function):
        subscribed = False
        is_union = False
        annotations = typing.get_type_hints(function)

        for name, event_class in annotations.items():
            origin = get_origin(event_class)
            events = []

            if origin is Union or isinstance(event_class, types.UnionType):
                logger.info(f"Parameter {name} is a Union: {event_class}")
                events = get_args(event_class)
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
                    logger.warning(f"Handler {function.__name__} registered for event {event_type}")
                elif not self._ignore_unknown_events:
                    raise KeyError(f"Event {sub_event} - {event_type} is not registered.")
                else:
                    logger.warning(f"Event {sub_event} - {event_type} - {type(sub_event)} {dir(sub_event)} is not registered – skipping handler {function.__name__}. All events: {self._events.keys()}")
        return function

    def _prepare_event(self, event):
        if isinstance(event, dict):
            event_type = event.get('type', '')
            try:
                event_class = self._events[event_type]
                event = event_class.from_dict(event, infer_missing=True)
            except Exception as exc:
                logger.exception(f"Can't convert dict {event} to event class, skipping")
                return

        if event.type in self._events:
            logger.info(f"Received event {event}")
            return event
        elif self._ignore_unknown_events:
                logger.info(f"Event not registered {event}")
        else:
            raise RuntimeError(f"Event not registered {event}")

    def append(self, *events):
        for event in events:
            event = self._prepare_event(event)
            if event:
                self._queue.append(event)

    async def send(self, *events):
        for event in reversed(events):
            event = self._prepare_event(event)
            if event:
                self._queue.appendleft(event)
        await self._process()

    async def _process(self):
        while self._queue:
            event = self._queue.popleft()
            for handler in self._handlers.get(event.type, []):
                try:
                    logger.info(f"Called handler {handler.__name__} for event {event.__class__.__name__}")
                    await handler(event)
                except Exception as exc:
                    self._queue.appendleft(ExceptionEvent(exc, handler))
                    logger.exception(f"Error calling handler {handler.__name__} for event {event.__class__.__name__}")

