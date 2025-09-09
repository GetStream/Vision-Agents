import asyncio
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user


async def main():
    client = AsyncClient()
    chat = client.chat.create(
        model="grok-3", messages=[system("You are a pirate assistant.")]
    )

    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        chat.append(user(prompt))
        response = await chat.sample()
        print(f"Grok: {response.content}")
        chat.append(response)


if __name__ == "__main__":
    asyncio.run(main())
