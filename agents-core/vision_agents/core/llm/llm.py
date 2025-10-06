from __future__ import annotations

import abc
import asyncio
import json
from typing import Optional, TYPE_CHECKING, Tuple, List, Dict, Any, TypeVar, Callable, Generic

from vision_agents.core.llm import events
from vision_agents.core.llm.events import ToolStartEvent, ToolEndEvent

if TYPE_CHECKING:
    from vision_agents.core.agents import Agent
    from vision_agents.core.agents.conversation import Conversation

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from vision_agents.core.processors import Processor
from vision_agents.core.utils.utils import parse_instructions
from vision_agents.core.events.manager import EventManager
from .function_registry import FunctionRegistry
from .llm_types import ToolSchema, NormalizedToolCallItem

T = TypeVar("T")


class LLMResponseEvent(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text


BeforeCb = Callable[[List[Any]], None]
AfterCb = Callable[[LLMResponseEvent], None]


class LLM(abc.ABC):
    # if we want to use realtime/ sts behaviour
    sts: bool = False

    before_response_listener: BeforeCb
    after_response_listener: AfterCb
    agent: Optional["Agent"]
    _conversation: Optional["Conversation"]
    function_registry: FunctionRegistry

    def __init__(self):
        super().__init__()
        self.agent = None
        self.events = EventManager()
        self.events.register_events_from_module(events)
        self.function_registry = FunctionRegistry()

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        raise NotImplementedError

    def _build_enhanced_instructions(self) -> Optional[str]:
        """
        Build enhanced instructions by combining the original instructions with markdown file contents.

        Returns:
            Enhanced instructions string with markdown file contents included, or None if no parsed instructions
        """
        if not hasattr(self, 'parsed_instructions') or not self.parsed_instructions:
            return None

        parsed = self.parsed_instructions
        enhanced_instructions = [parsed.input_text]

        # Add markdown file contents if any exist
        if parsed.markdown_contents:
            enhanced_instructions.append("\n\n## Referenced Documentation:")
            for filename, content in parsed.markdown_contents.items():
                if content:  # Only include non-empty content
                    enhanced_instructions.append(f"\n### {filename}")
                    enhanced_instructions.append(content)
                else:
                    enhanced_instructions.append(f"\n### {filename}")
                    enhanced_instructions.append("*(File not found or could not be read)*")

        return "\n".join(enhanced_instructions)

    def _get_tools_for_provider(self) -> List[Dict[str, Any]]:
        """
        Get tools in provider-specific format.
        This method should be overridden by each LLM implementation.
        
        Returns:
            List of tools in the provider's expected format.
        """
        tools = self.get_available_functions()
        return self._convert_tools_to_provider_format(tools)
    
    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to provider-specific format.
        This method should be overridden by each LLM implementation.
        
        Args:
            tools: List of ToolSchema objects
            
        Returns:
            List of tools in provider-specific format
        """
        # Default implementation - should be overridden
        return []
    
    def _extract_tool_calls_from_response(self, response: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from provider-specific response.
        This method should be overridden by each LLM implementation.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            List of normalized tool call items
        """
        # Default implementation - should be overridden
        return []
    
    def _extract_tool_calls_from_stream_chunk(self, chunk: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from a streaming chunk.
        This method should be overridden by each LLM implementation.
        
        Args:
            chunk: Provider-specific streaming chunk
            
        Returns:
            List of normalized tool call items
        """
        # Default implementation - should be overridden
        return []
    
    def _create_tool_result_message(self, tool_calls: List[NormalizedToolCallItem], results: List[Any]) -> List[Dict[str, Any]]:
        """
        Create tool result messages for the provider.
        This method should be overridden by each LLM implementation.
        
        Args:
            tool_calls: List of tool calls that were executed
            results: List of results from function execution
            
        Returns:
            List of tool result messages in provider format
        """
        # Default implementation - should be overridden
        return []

    def _attach_agent(self, agent: Agent):
        """
        Attach agent to the llm
        """
        self.agent = agent
        self._conversation = agent.conversation
        self.instructions = agent.instructions
        
        # Parse instructions to extract @ mentioned markdown files
        self.parsed_instructions = parse_instructions(agent.instructions)

    def register_function(self, 
                         name: Optional[str] = None,
                         description: Optional[str] = None) -> Callable:
        """
        Decorator to register a function with the LLM's function registry.
        
        Args:
            name: Optional custom name for the function. If not provided, uses the function name.
            description: Optional description for the function. If not provided, uses the docstring.
        
        Returns:
            Decorator function.
        """
        return self.function_registry.register(name, description)
    
    def get_available_functions(self) -> List[ToolSchema]:
        """Get a list of available function schemas."""
        return self.function_registry.get_tool_schemas()
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a registered function with the given arguments.
        
        Args:
            name: Name of the function to call.
            arguments: Dictionary of arguments to pass to the function.
        
        Returns:
            Result of the function call.
        """
        return self.function_registry.call_function(name, arguments)
    

    def _tc_key(self, tc: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
        """Generate a unique key for tool call deduplication.
        
        Args:
            tc: Tool call dictionary
            
        Returns:
            Tuple of (id, name, arguments_json) for deduplication
        """
        return (
            tc.get("id"), 
            tc["name"], 
            json.dumps(tc.get("arguments_json", tc.get("arguments", {})), sort_keys=True)
        )

    async def _maybe_await(self, x):
        """Await if x is a coroutine, otherwise return x directly.
        
        Args:
            x: Value that might be a coroutine
            
        Returns:
            Awaited result if coroutine, otherwise x
        """
        if asyncio.iscoroutine(x):
            return await x
        return x

    async def _run_one_tool(self, tc: Dict[str, Any], timeout_s: float):
        """Run a single tool call with timeout.
        
        Args:
            tc: Tool call dictionary
            timeout_s: Timeout in seconds
            
        Returns:
            Tuple of (tool_call, result, error)
        """
        import inspect
        import time
        
        args = tc.get("arguments_json", tc.get("arguments", {})) or {}
        start_time = time.time()
        
        async def _invoke():
            # Get the actual function to check if it's async
            if hasattr(self.function_registry, 'get_callable'):
                fn = self.function_registry.get_callable(tc["name"])
                if inspect.iscoroutinefunction(fn):
                    return await fn(**args)
                else:
                    # Run sync function in a worker thread to avoid blocking
                    return await asyncio.to_thread(fn, **args)
            else:
                # Fallback to existing call_function method
                res = self.call_function(tc["name"], args)
                return await self._maybe_await(res)
        
        try:
            # Send tool start event
            self.events.send(ToolStartEvent(
                plugin_name="llm",
                tool_name=tc["name"],
                arguments=args,
                tool_call_id=tc.get("id")
            ))
            
            res = await asyncio.wait_for(_invoke(), timeout=timeout_s)
            execution_time = (time.time() - start_time) * 1000
            
            # Send tool end event (success)
            self.events.send(ToolEndEvent(
                plugin_name="llm",
                tool_name=tc["name"],
                success=True,
                result=res,
                tool_call_id=tc.get("id"),
                execution_time_ms=execution_time
            ))
            
            return tc, res, None
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Send tool end event (error)
            self.events.send(ToolEndEvent(
                plugin_name="llm",
                tool_name=tc["name"],
                success=False,
                error=str(e),
                tool_call_id=tc.get("id"),
                execution_time_ms=execution_time
            ))
            
            return tc, {"error": str(e)}, e

    async def _execute_tools(self, calls: List[Dict[str, Any]], *, max_concurrency: int = 8, timeout_s: float = 30):
        """Execute multiple tool calls concurrently with timeout.
        
        Args:
            calls: List of tool call dictionaries
            max_concurrency: Maximum number of concurrent tool executions
            timeout_s: Timeout per tool execution in seconds
            
        Returns:
            List of tuples (tool_call, result, error)
        """
        sem = asyncio.Semaphore(max_concurrency)
        
        async def _guarded(tc):
            async with sem:
                return await self._run_one_tool(tc, timeout_s)
        
        return await asyncio.gather(*[_guarded(tc) for tc in calls])

    async def _dedup_and_execute(
        self,
        calls: List[Dict[str, Any]],
        *,
        max_concurrency: int = 8,
        timeout_s: float = 30,
        seen: Optional[set] = None,
    ):
        """De-duplicate (by id/name/args) then execute concurrently.
        
        Args:
            calls: List of tool call dictionaries
            max_concurrency: Maximum number of concurrent tool executions
            timeout_s: Timeout per tool execution in seconds
            seen: Set of seen tool call keys for deduplication
            
        Returns:
            Tuple of (triples, updated_seen_set)
        """
        seen = seen or set()
        to_run: List[Dict[str, Any]] = []
        for tc in calls:
            key = self._tc_key(tc)
            if key in seen:
                continue
            seen.add(key)
            to_run.append(tc)

        if not to_run:
            return [], seen  # nothing new

        triples = await self._execute_tools(to_run, max_concurrency=max_concurrency, timeout_s=timeout_s)
        return triples, seen

    def _sanitize_tool_output(self, value: Any, max_chars: int = 60_000) -> str:
        """Sanitize tool output to prevent oversized responses.
        
        Args:
            value: Tool output value
            max_chars: Maximum characters allowed
            
        Returns:
            Sanitized string output
        """
        s = value if isinstance(value, str) else json.dumps(value)
        return (s[:max_chars] + "…") if len(s) > max_chars else s
