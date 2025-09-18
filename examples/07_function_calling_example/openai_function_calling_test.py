import asyncio
import sys
import os
from dotenv import load_dotenv

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "agents-core"))
sys.path.insert(0, os.path.join(project_root, "plugins"))

load_dotenv()

from plugins.openai.stream_agents.plugins.openai.openai_llm import OpenAILLM

async def main():
    print("🚀 OpenAI Function Calling Test")
    print("=" * 50)
    
    # Initialize LLM
    llm = OpenAILLM(model="gpt-4o")
    print("✅ OpenAILLM initialized")
    
    # Register test functions
    @llm.register_function(description="Get current weather for a location")
    def get_weather(location: str):
        """Get weather information for a specific location."""
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "sunny",
            "humidity": "65%",
            "wind": "5 mph"
        }
    
    @llm.register_function(description="Calculate the square of a number")
    def square_number(number: int):
        """Calculate the square of a given number."""
        return {"number": number, "square": number * number}
    
    @llm.register_function(description="Get user information by ID")
    def get_user_info(user_id: str):
        """Get user information from the database."""
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "status": "active"
        }
    
    print("✅ Functions registered successfully")
    print(f"📋 Available functions: {len(llm.get_available_functions())}")
    
    # Test 1: Simple function call
    print("\n🔧 Test 1: Simple function call")
    print("-" * 30)
    try:
        response = await llm.create_response(
            "What's the weather in Tokyo?",
            stream=False
        )
        print(f"✅ Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test 2: Multiple function calls
    print("\n🔧 Test 2: Multiple function calls")
    print("-" * 30)
    try:
        response = await llm.create_response(
            "What's the weather in London and what's 7 squared?",
            stream=False
        )
        print(f"✅ Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test 3: Complex query with multiple functions
    print("\n🔧 Test 3: Complex query")
    print("-" * 30)
    try:
        response = await llm.create_response(
            "I need to know the weather in Paris, calculate 9 squared, and get info for user '12345'",
            stream=False
        )
        print(f"✅ Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test 4: Streaming function calls
    print("\n🔧 Test 4: Streaming function calls")
    print("-" * 30)
    try:
        response = await llm.create_response(
            "What's the weather in New York and calculate 12 squared?",
            stream=True
        )
        print(f"✅ Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n🎉 All tests completed successfully!")
    print("✅ Function calling is working perfectly after cleanup!")

if __name__ == "__main__":
    asyncio.run(main())
