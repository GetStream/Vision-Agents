import logging
import openai
import fastmcp
import json
import re

logging.basicConfig(level=logging.INFO)


SYSTEM = """
You are a friendly AI assistant.  When the user asks for real-world data, look through the
available MCP tools and call the one that matches. For weather, if you see the degrees symbol, replace it with the word "degrees" in your response. Reply with plain text.
If you call a tool, use  "tool_name"["argument_key": "argument_value"]  on its own line.
You can make up to 3 tool calls in a row.
"""


def call_llm(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    content = response.choices[0].message.content
    logging.info(f"LLM content: {content}")
    return content


async def chat_with_tools(prompt: str, client: fastmcp.Client) -> str:
    tools = await client.list_tools()
    tool_names = {t.name for t in tools}
    tool_descriptions = [f"{t.name}: {t.description}" for t in tools]
    history = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "Available tools:\n" + "\n".join(tool_descriptions),
        },
    ]

    tool_calls = 0

    while True:
        reply = call_llm(history)
        m = re.match(r"^(\w[\w\.]+)\[(.*)\]$", reply.strip(), re.DOTALL)

        # If the LLM calls a tool, invoke it and add both the call and the result to the history
        if m and m.group(1) in tool_names:
            tool_name, arg_json = m.groups()

            # Record the assistant's tool call message for grounding
            history.append({"role": "assistant", "content": reply})

            # Parse arguments robustly
            try:
                args = json.loads("{" + arg_json + "}")
            except Exception as e:
                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous tool call arguments were not valid JSON (error: {e}). "
                            f'Please retry using a valid JSON object inside the brackets, e.g. {tool_name}["param":"value"].'
                        ),
                    }
                )
                continue

            # Call the tool with error handling
            try:
                result = await client.call_tool(tool_name, args)
                history.append(
                    {"role": "assistant", "content": f"(result) {result.data}"}
                )
            except Exception as e:
                history.append(
                    {
                        "role": "assistant",
                        "content": f"(error) Tool '{tool_name}' failed: {e}",
                    }
                )

            tool_calls += 1
            if tool_calls == 2:
                history.append(
                    {
                        "role": "user",
                        "content": "You may make at most one more tool call. If not strictly necessary, answer the user directly now.",
                    }
                )
            if tool_calls >= 3:
                history.append(
                    {
                        "role": "user",
                        "content": "Do not call more tools. Answer the user's question directly now using the information you have.",
                    }
                )
                final = call_llm(history)
                return final

            continue
        else:
            return reply
