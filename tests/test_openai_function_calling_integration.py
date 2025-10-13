import os
import pytest
import dotenv

from vision_agents.plugins.openai import LLM as OpenAILLM


dotenv.load_dotenv()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_function_calling_live_roundtrip():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping live integration test")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Use live OpenAI client (api key taken from env)
    llm = OpenAILLM(model=model)

    # Stub event emitter to avoid relying on event infra
    setattr(llm, "emit", lambda *args, **kwargs: None)

    # Side-effect to prove the tool actually ran
    calls: list[str] = []

    @llm.register_function(
        description="Probe tool that records invocation and returns a marker string"
    )
    def probe_tool(ping: str) -> str:
        calls.append(ping)
        return f"probe_ok:{ping}"

    # Strongly nudge the model to use the tool deterministically
    prompt = (
        "You MUST call the tool named 'probe_tool' with the parameter ping='pong' now. "
        "After receiving the tool result, reply by returning ONLY the tool result string and nothing else."
    )

    res = await llm.create_response(input=prompt)

    # Assert the tool was executed (side-effect) and we got a text back
    assert len(calls) >= 1, "probe_tool was not invoked by the model"
    assert isinstance(res.text, str)
    # Prefer that the model echoes the tool result; this verifies follow-up worked
    assert "probe_ok:pong" in res.text
