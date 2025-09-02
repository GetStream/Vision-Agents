"""
Base LLM classes with function calling support.

Requirements
- support image, text, functools etc as input

This module provides the base classes for LLM implementations with
automatic function calling capabilities.
"""

from typing import List, Dict, Any, Optional, Union, AsyncIterator
import logging
from .function_registry import FunctionRegistry



class LLM:
    """
    Base class for LLM implementations with function calling support.
    
    All LLM implementations should inherit from this class to get
    automatic function calling capabilities.
    """
    
    def __init__(self):
        self.function_registry = FunctionRegistry()
        self.sts: bool = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def function(self, description: str = "", name: Optional[str] = None):
        """
        Decorator to register a function for LLM calling.
        
        Args:
            description: Human-readable description of what the function does
            name: Optional custom name for the function (defaults to function name)
        
        Example:
            @llm.function("Get weather for a location")
            async def get_weather(location: str) -> str:
                return f"Weather in {location}: sunny"
        """
        return self.function_registry.function(description, name)
    
    async def generate_with_functions(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Generate response with function calling support.
        
        This method should be implemented by each LLM subclass to handle
        function calling in a provider-specific way.
        """
        raise NotImplementedError("Subclasses must implement generate_with_functions")
    
    def get_available_functions(self) -> List[str]:
        """Get list of available function names."""
        return self.function_registry.list_functions()
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered function."""
        return self.function_registry.get_function_info(name)



class RealtimeLLM(LLM):
    """
    Base class for real-time LLM implementations (Speech-to-Speech).
    
    These LLMs handle audio directly and support real-time conversation.
    """
    
    def __init__(self):
        super().__init__()
        self.sts: bool = True
    
    async def connect(self, call, agent_user_id: str = "assistant"):
        """
        Connect to a video call for real-time communication.
        
        This method should be implemented by each real-time LLM subclass.
        """
        raise NotImplementedError("Subclasses must implement connect")
    
    def attach_incoming_audio(self, track):
        """Attach incoming audio track for processing."""
        raise NotImplementedError("Subclasses must implement attach_incoming_audio")
    
    def attach_outgoing_audio(self, track):
        """Attach outgoing audio track for responses."""
        raise NotImplementedError("Subclasses must implement attach_outgoing_audio")

    def attach_outgoing_audio(self, track):
        pass
