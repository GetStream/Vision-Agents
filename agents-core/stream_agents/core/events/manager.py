import collections
import dataclasses
import types
import logging


logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ExceptionEvent:
    exc: Exception
    handler: types.FunctionType
    type: str = 'event.exception'


class EventManager:
    def __init__(self):
        self._queue = collections.deque([])
        self._events = {}
        self._handlers = {}
        self._modules = {}

        self.register(ExceptionEvent)

    def register(self, event_class):
        if event_class.__name__.endswith('Event') and hasattr(event_class, 'type'):
            self._events[event_class.type] = event_class
        else:
            raise ValueError("Provide valid class that ends on '*Event' and 'type' attribute")


    def register_events_from_module(self, module, prefix):
        for name, class_ in module.__dict__.items():
            if name.endswith('Event') and getattr(class_, 'type', '').startswith(prefix):
                self.register(class_)
                self._modules.setdefault(module, []).append(class_)

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

    def listen(self, function):
        subscribed = False
        for name, event_class in function.__annotations__.items():
            # check union of event classes (not neeeded for now)
            if type(event_class) is types.UnionType or subscribed:
                raise ValueError("Multiple events per handler don't supported")
            if event_class.type in self._events:
                subscribed = True
                self._handlers.setdefault(event_class.type, []).append(function)
            else:
                raise KeyError(f"Event {event_class} is not registered.")
        return function

    def _prepare_event(self, event):
        if isinstance(event, dict):
            event_type = event.get('type', '')
            if event_type in self._events:
                event_class = self._events[event_type]
                try:
                    event = event_class.from_dict(event, infer_missing=True)
                except Exception as exc:
                    logger.exception(f"Can't convert {event_class} from {event}")
                    return

                logger.info(f"Received event {event}")
                return event
            else:
                logger.info(f"Event not registered {event}")
        elif event.type not in self._events:
            logger.info(f"Event not registered {event}")

    async def append(self, *events):
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
                    await handler(event)
                except Exception as exc:
                    self._queue.appendleft(ExceptionEvent(exc, handler))

