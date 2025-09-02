"""
Simple Custom Agent Example

This is a minimal example showing how to create a custom agent with function calling.
"""

import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from getstream import Stream
from getstream.plugins import DeepgramSTT, ElevenLabsTTS
from stream_agents.turn_detection import FalTurnDetection
from stream_agents.llm import OpenAILLM
from stream_agents import Agent, StreamEdge, start_dispatcher, open_demo


class SimpleTaskAgent(Agent):
    """
    A simple custom agent that manages a basic task list.
    """
    
    def __init__(self, **kwargs):
        # Initialize the base agent
        super().__init__(**kwargs)
        
        # Initialize custom state
        self.tasks = []
        
        # Set custom instructions
        if self.llm and hasattr(self.llm, 'instructions'):
            self.llm.instructions = """You are a simple task management assistant. 
            You can help users add, list, and complete tasks. Be friendly and helpful."""
        
        # Register the task management functions
        self._register_task_functions()
        
        print("✅ Simple Task Agent initialized")
    
    def _register_task_functions(self):
        """Register task management functions."""
        
        @self.llm.function("Add a new task")
        def add_task(task: str) -> str:
            """Add a new task to the list."""
            task_id = len(self.tasks) + 1
            self.tasks.append({
                "id": task_id,
                "task": task,
                "completed": False
            })
            return f"✅ Added task: '{task}' (ID: {task_id})"
        
        @self.llm.function("List all tasks")
        def list_tasks() -> str:
            """List all tasks in the task list."""
            if not self.tasks:
                return "📝 No tasks found. You're all caught up!"
            
            task_list = []
            for task in self.tasks:
                status = "✅" if task["completed"] else "⏳"
                task_list.append(f"{status} {task['id']}. {task['task']}")
            
            return "📝 Your tasks:\n" + "\n".join(task_list)
        
        @self.llm.function("Complete a task")
        def complete_task(task_id: int) -> str:
            """Mark a task as completed."""
            for task in self.tasks:
                if task["id"] == task_id:
                    if task["completed"]:
                        return f"⚠️ Task {task_id} is already completed."
                    task["completed"] = True
                    return f"✅ Completed task: '{task['task']}'"
            
            return f"❌ Task with ID {task_id} not found."
        
        @self.llm.function("Get task count")
        def get_task_count() -> str:
            """Get the number of pending tasks."""
            pending = len([t for t in self.tasks if not t["completed"]])
            total = len(self.tasks)
            return f"📊 You have {pending} pending tasks out of {total} total tasks."


async def main():
    """Main function to run the simple custom agent example."""
    
    print("🤖 Simple Custom Agent Example")
    print("=" * 40)
    
    load_dotenv()
    
    # Create Stream client
    client = Stream.from_env()
    
    # Create the custom agent
    agent = SimpleTaskAgent(
        edge=StreamEdge(),
        llm=OpenAILLM(name="gpt-4o-mini"),  # Use cheaper model for demo
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        turn_detection=FalTurnDetection(),
    )
    
    # Create a call
    call = client.video.call("default", str(uuid4()))
    
    # Open the demo UI
    open_demo(call)
    
    # Have the agent join the call
    with await agent.join(call):
        # Send initial greeting
        await agent.create_response("Hello! I'm your task management assistant. I can help you add tasks, list them, and mark them as completed. What would you like to do?")
        
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_dispatcher(main))
