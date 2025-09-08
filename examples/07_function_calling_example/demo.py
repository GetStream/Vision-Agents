#!/usr/bin/env python3
"""
Simple demonstration of function calling capabilities.
This script shows how to register and use functions with the LLM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents-core'))

from stream_agents.core.llm import FunctionRegistry


def main():
    """Demonstrate function calling without requiring API keys."""
    
    # Create a function registry
    registry = FunctionRegistry()
    
    # Register some example functions
    @registry.register(description="Get current weather for a location")
    def get_weather(location: str) -> dict:
        """Get the current weather for a location."""
        # Mock weather data
        weather_data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%"
        }
        return weather_data
    
    @registry.register(description="Calculate the sum of two numbers")
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    @registry.register(description="Get user information by ID")
    def get_user_info(user_id: str) -> dict:
        """Get information about a user by their ID."""
        # Mock user data
        user_data = {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "status": "active"
        }
        return user_data
    
    @registry.register(description="Convert temperature between Celsius and Fahrenheit")
    def convert_temperature(temp: float, from_unit: str, to_unit: str) -> dict:
        """Convert temperature between Celsius and Fahrenheit."""
        if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
            converted = (temp * 9/5) + 32
        elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
            converted = (temp - 32) * 5/9
        else:
            raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")
        
        return {
            "original_temp": temp,
            "original_unit": from_unit,
            "converted_temp": round(converted, 2),
            "converted_unit": to_unit
        }
    
    print("Function Calling Demo")
    print("=" * 50)
    
    # Show available functions
    print("\nAvailable functions:")
    for schema in registry.get_tool_schemas():
        print(f"- {schema['name']}: {schema.get('description', 'No description')}")
        print(f"  Parameters: {list(schema['parameters_schema']['properties'].keys())}")
    
    print("\n" + "=" * 50)
    print("Testing function calls:")
    print("=" * 50)
    
    # Test different function calls
    test_calls = [
        ("get_weather", {"location": "New York"}),
        ("calculate_sum", {"a": 15, "b": 27}),
        ("get_user_info", {"user_id": "12345"}),
        ("convert_temperature", {"temp": 25, "from_unit": "celsius", "to_unit": "fahrenheit"}),
    ]
    
    for func_name, args in test_calls:
        print(f"\nCalling {func_name}({', '.join(f'{k}={v}' for k, v in args.items())})")
        try:
            result = registry.call_function(func_name, args)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing error handling:")
    print("=" * 50)
    
    # Test error cases
    error_cases = [
        ("nonexistent_function", {"arg": "value"}),
        ("calculate_sum", {"a": 5}),  # Missing required parameter
        ("convert_temperature", {"temp": 25, "from_unit": "celsius", "to_unit": "kelvin"}),  # Invalid conversion
    ]
    
    for func_name, args in error_cases:
        print(f"\nCalling {func_name}({', '.join(f'{k}={v}' for k, v in args.items())})")
        try:
            result = registry.call_function(func_name, args)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Function schemas (JSON format):")
    print("=" * 50)
    
    import json
    for schema in registry.get_tool_schemas():
        print(f"\n{schema['name']}:")
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
