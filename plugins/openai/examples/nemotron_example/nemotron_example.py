"""
NVIDIA Nemotron Example — Fraud Detection Demo

Demonstrates NVIDIA Nemotron hosted on Baseten with tool calling
in a fraud detection scenario.

Creates a voice agent that uses:
- OpenAI ChatCompletions LLM pointed at Baseten's inference API (Nemotron)
- Tool calling to look up transactions and flag fraud
- send_custom_event to share transaction data with call participants
- Deepgram for speech-to-text (STT) and text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- BASETEN_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem
from vision_agents.plugins import deepgram, getstream, openai

logger = logging.getLogger(__name__)

load_dotenv()

TOOL_CALLS_LOG = Path("tool_calls.jsonl")


def _log_tool_call(name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Append a tool call record to tool_calls.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": name,
        "arguments": args,
        "result": result,
    }
    with TOOL_CALLS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


THINK_END_TAG = "</think>"
TOOL_CALL_TAG = "<tool_call>"
PLUGIN_NAME = "chat_completions_llm"
MAX_RESPONSE_TOKENS = 4096
_MARKDOWN_RE = re.compile(r"[\*_#`~]")
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)

SUMMARIZE_PROMPT = (
    "Rewrite the following as something a real person would say on the phone — casual, warm, natural. "
    "Two to three short sentences, around 25-35 words. No jargon, no IDs, no markdown, no em-dashes, no bullet points.\n\n"
    "Keep all key info about actions taken (frozen card, cancelled charge, replacement card, refund timeline). "
    "Example tone: 'Okay so I've frozen your card and cancelled that charge. The money should be back in a couple of days. Want me to send you a new card?'\n\n"
    "Do not use any special characters that only make sense in writing like brackets.\n"

    "here's the prompt i gave to the llm that generates the text you review:\n\n"

"""    INSTRUCTIONS = (
    "detailed thinking off\n\n"
    "You are a bank fraud phone agent on a live phone call. Your output goes directly to TTS.\n\n"
    "When you spot a suspicious transaction, explain WHY — e.g. 'There's a large charge in Miami but you live in London.' "
    "Suspicious means: different city from the customer's home, or unusually large compared to their other transactions.\n"
    "If you flag a transaction, the card should be frozen immediately to prevent further fraud, but only after you have explicitly confirmed with the customer that they did not make the transaction.\n\n"
    "If you freeze a card due to a transaction, you should also offer to issue a replacement card. If the customer confirms, issue a virtual card immediately and a physical card that arrives in 3-5 business days.\n\n"
    "If a transaction is fraudulent, you should also cancel the charge to return the money to the customer's account. This typically takes 24-48 hours.\n\n"
    "The user cannot see what you can see. They are on the phone."
    "RULES:\n"
    "- MAXIMUM 15 words per response. One short sentence. Responses over 15 words get cut off.\n"
    "- NEVER take action (flag, freeze, cancel) without the customer explicitly confirming.\n"
    "- After taking any action, always tell the customer what you did.\n"
    "- Work one step at a time: look up info, briefly tell the customer what you found, ask what to do.\n"
    "- Do NOT list data, read IDs, dates, or dollar amounts aloud.\n"
    "- Speak casually like a real person. No markdown, no bullet points.\n"
)
"""
)

_summarizer = AsyncOpenAI()


async def _summarize_for_speech(text: str) -> str:
    """Use GPT-4o-mini to condense a verbose response into one spoken sentence."""
    response = await _summarizer.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SUMMARIZE_PROMPT + text}],
        max_tokens=120,
        temperature=0.3,
    )
    result = response.choices[0].message.content or text
    return result.strip().strip('"')

MOCK_TRANSACTIONS = [
    {"id": "tx_001", "amount": 7.50, "merchant": "Greg's Questionable Tacos", "city": "London", "date": "2026-03-12"},
    {"id": "tx_002", "amount": 42.00, "merchant": "The Existential Bookshop", "city": "London", "date": "2026-03-11"},
    {"id": "tx_003", "amount": 8900.00, "merchant": "Luxe Diamond Emporium", "city": "Miami", "date": "2026-03-12"},
]

MOCK_ACCOUNT = {
    "account_id": "1234",
    "name": "Alex Johnson",
    "home_city": "London",
    "card_status": "active",
}

INSTRUCTIONS = (
    "detailed thinking off\n\n"
    "You are a bank fraud phone agent on a live phone call. Your output goes directly to TTS.\n\n"
    "When you spot a suspicious transaction, explain WHY — e.g. 'There's a large charge in Miami but you live in London.' "
    "Suspicious means: different city from the customer's home, or unusually large compared to their other transactions.\n"
    "If you flag a transaction, the card should be frozen immediately to prevent further fraud, but only after you have explicitly confirmed with the customer that they did not make the transaction.\n\n"
    "If you freeze a card due to a transaction, you should also offer to issue a replacement card. If the customer confirms, issue a virtual card immediately and a physical card that arrives in 3-5 business days.\n\n"
    "If a transaction is fraudulent, you should also cancel the charge to return the money to the customer's account. This typically takes 24-48 hours.\n\n"
    "The user cannot see what you can see. They are on the phone."
    "RULES:\n"
    "- MAXIMUM 15 words per response. One short sentence. Responses over 15 words get cut off.\n"
    "- NEVER take action (flag, freeze, cancel) without the customer explicitly confirming.\n"
    "- After taking any action, always tell the customer what you did.\n"
    "- Work one step at a time: look up info, briefly tell the customer what you found, ask what to do.\n"
    "- Do NOT list data, read IDs, dates, or dollar amounts aloud.\n"
    "- Speak casually like a real person. No markdown, no bullet points.\n"
)


class NemotronLLM(openai.ChatCompletionsLLM):
    """ChatCompletionsLLM that handles Nemotron's thinking and XML tool calls.

    Nemotron emits reasoning text ending with </think>, and outputs tool calls as
    XML (<tool_call><function=name><parameter=key>value</parameter></function></tool_call>)
    instead of using the native OpenAI tool_calls format.
    """

    async def _create_response_internal(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Override to inject max_tokens for short responses."""
        tool_depth = kwargs.get("_tool_depth", 0)
        if tool_depth >= 5:
            tools = None

        request_kwargs: Dict[str, Any] = {
            "messages": messages,
            "model": kwargs.get("model", self.model),
            "stream": stream,
            "max_tokens": MAX_RESPONSE_TOKENS,
        }
        if tools:
            request_kwargs["tools"] = tools

        request_start_time = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception:
            logger.exception("Failed to get a response from Nemotron")
            return LLMResponseEvent(original=None, text="")

        return await self._process_streaming_response(
            response, messages, tools, kwargs, request_start_time
        )

    async def _process_streaming_response(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
        request_start_time: float,
    ) -> LLMResponseEvent:
        """Collect response, strip thinking, and handle XML tool calls."""
        full_text = ""
        last_chunk: Optional[ChatCompletionChunk] = None
        first_token_time: Optional[float] = None

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += chunk.choices[0].delta.content
            last_chunk = chunk

        # Strip everything up to and including </think>
        think_end = full_text.find(THINK_END_TAG)
        if think_end != -1:
            print(f">>> THINK TAG FOUND at pos {think_end} / {len(full_text)} chars")
            full_text = full_text[think_end + len(THINK_END_TAG) :].lstrip()
            print(f">>> AFTER STRIP: {full_text[:200]}")
        else:
            print(f">>> NO THINK TAG. Raw: {full_text[:300]}")

        # Check for XML tool calls (max 5 rounds of tool calls per user turn)
        tool_depth = kwargs.get("_tool_depth", 0)
        tool_calls = _parse_nemotron_tool_calls(full_text)
        if tool_calls and tool_depth < 5:
            return await self._execute_xml_tool_calls(
                tool_calls, full_text, messages, tools, kwargs
            )

        # Strip tool call XML and markdown, then summarize for TTS
        full_text = _TOOL_CALL_RE.sub("", full_text)
        full_text = _MARKDOWN_RE.sub("", full_text)
        full_text = " ".join(full_text.split())
        if full_text and len(full_text.split()) > 40:
            full_text = await _summarize_for_speech(full_text)

        # Normal text response — emit events for TTS
        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_ms = (first_token_time - request_start_time) * 1000 if first_token_time else None
        item_id = last_chunk.id if last_chunk else None

        if full_text:
            self.events.send(
                LLMResponseChunkEvent(
                    plugin_name=PLUGIN_NAME,
                    content_index=None,
                    item_id=item_id,
                    output_index=0,
                    sequence_number=0,
                    delta=full_text,
                    is_first_chunk=True,
                    time_to_first_token_ms=ttft_ms,
                )
            )

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                original=last_chunk,
                text=full_text,
                item_id=item_id,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                model=self.model,
            )
        )

        return LLMResponseEvent(original=last_chunk, text=full_text)

    async def _execute_xml_tool_calls(
        self,
        tool_calls: List[NormalizedToolCallItem],
        tool_call_text: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> LLMResponseEvent:
        """Execute parsed XML tool calls and get a follow-up response."""
        triples, _ = await self._dedup_and_execute(
            tool_calls, max_concurrency=8, timeout_s=30
        )
        if not triples:
            return LLMResponseEvent(original=None, text="")

        current_messages = list(messages)
        current_messages.append({"role": "assistant", "content": tool_call_text})

        results = []
        for tc, result, error in triples:
            output = error if error is not None else result
            results.append(json.dumps({"name": tc["name"], "content": output}))

        current_messages.append({
            "role": "user",
            "content": "<tool_response>\n" + "\n".join(results) + "\n</tool_response>",
        })

        follow_up_kwargs = {**kwargs, "_tool_depth": kwargs.get("_tool_depth", 0) + 1}
        return await self._create_response_internal(
            messages=current_messages, tools=tools, stream=True, **follow_up_kwargs
        )


def _parse_nemotron_tool_calls(text: str) -> list[NormalizedToolCallItem]:
    """Parse Nemotron's XML tool call format.

    Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    """
    tool_calls: list[NormalizedToolCallItem] = []
    for tc_match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        tc_body = tc_match.group(1)
        fn_match = re.search(r"<function=(\w+)>", tc_body)
        if not fn_match:
            continue
        params: dict[str, str] = {}
        for param_match in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", tc_body, re.DOTALL
        ):
            params[param_match.group(1)] = param_match.group(2).strip()
        tool_calls.append({
            "type": "tool_call",
            "id": str(uuid.uuid4()),
            "name": fn_match.group(1),
            "arguments_json": params,
        })
    return tool_calls


async def create_agent(**kwargs) -> Agent:
    """Create the agent with NVIDIA Nemotron via Baseten's inference API."""
    TOOL_CALLS_LOG.write_text("")
    tool_cache: dict[str, Dict[str, Any]] = {}

    llm = NemotronLLM(
        model="nvidia/Nemotron-120B-A12B",
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Fraud Detection Agent", id="agent"),
        instructions=INSTRUCTIONS,
        llm=llm,
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
    )

    @llm.register_function(description="Get the customer's account information including name, home city, and card status")
    async def get_account_info(account_id: str) -> Dict[str, Any]:
        cache_key = f"get_account_info:{account_id}"
        if cache_key in tool_cache:
            logger.info("🔍 get_account_info(account_id=%s) — cached", account_id)
            return tool_cache[cache_key]
        logger.info("🔍 Tool call: get_account_info(account_id=%s)", account_id)
        await agent.send_custom_event({"type": "account_info", "data": MOCK_ACCOUNT})
        logger.info("🔍 Account info: %s (%s)", MOCK_ACCOUNT["name"], MOCK_ACCOUNT["home_city"])
        _log_tool_call("get_account_info", {"account_id": account_id}, MOCK_ACCOUNT)
        tool_cache[cache_key] = MOCK_ACCOUNT
        return MOCK_ACCOUNT

    @llm.register_function(description="Get recent transactions for the customer's account")
    async def get_recent_transactions(account_id: str) -> Dict[str, Any]:
        cache_key = f"get_recent_transactions:{account_id}"
        if cache_key in tool_cache:
            logger.info("🔍 get_recent_transactions(account_id=%s) — cached", account_id)
            return tool_cache[cache_key]
        logger.info("🔍 Tool call: get_recent_transactions(account_id=%s)", account_id)
        await agent.send_custom_event({"type": "transactions", "data": MOCK_TRANSACTIONS})
        logger.info("🔍 Found %d transactions", len(MOCK_TRANSACTIONS))
        result = {"transactions": MOCK_TRANSACTIONS}
        _log_tool_call("get_recent_transactions", {"account_id": account_id}, result)
        tool_cache[cache_key] = result
        return result

    @llm.register_function(description="Flag a transaction as fraudulent and freeze the card")
    async def flag_transaction(transaction_id: str, reason: str) -> Dict[str, Any]:
        cache_key = f"flag_transaction:{transaction_id}"
        if cache_key in tool_cache:
            logger.info("🚨 flag_transaction(id=%s) — already done", transaction_id)
            return {**tool_cache[cache_key], "status": "already_flagged"}
        logger.info("🚨 Tool call: flag_transaction(id=%s, reason=%s)", transaction_id, reason)
        result = {
            "status": "flagged",
            "transaction_id": transaction_id,
            "reason": reason,
            "card_frozen": True,
        }
        await agent.send_custom_event({"type": "transaction_flagged", "data": result})
        logger.info("🚨 Transaction %s flagged — card frozen", transaction_id)
        _log_tool_call("flag_transaction", {"transaction_id": transaction_id, "reason": reason}, result)
        tool_cache[cache_key] = result
        return result

    @llm.register_function(description="Issue a replacement card and return the new card details")
    async def issue_replacement_card(account_id: str) -> Dict[str, Any]:
        cache_key = f"issue_replacement_card:{account_id}"
        if cache_key in tool_cache:
            logger.info("💳 issue_replacement_card(account_id=%s) — already done", account_id)
            return {**tool_cache[cache_key], "status": "already_issued"}
        logger.info("💳 Tool call: issue_replacement_card(account_id=%s)", account_id)
        result = {
            "status": "issued",
            "card_last_four": "7821",
            "card_type": "virtual",
            "delivery": "instant",
            "physical_card_eta": "3-5 business days",
        }
        await agent.send_custom_event({"type": "replacement_card_issued", "data": result})
        logger.info("💳 Virtual card issued ending in 7821")
        _log_tool_call("issue_replacement_card", {"account_id": account_id}, result)
        tool_cache[cache_key] = result
        return result

    @llm.register_function(description="Cancel a fraudulent charge and return the money to the customer's account")
    async def cancel_charge(transaction_id: str) -> Dict[str, Any]:
        cache_key = f"cancel_charge:{transaction_id}"
        if cache_key in tool_cache:
            logger.info("💰 cancel_charge(id=%s) — already done", transaction_id)
            return {**tool_cache[cache_key], "status": "already_cancelled"}
        logger.info("💰 Tool call: cancel_charge(transaction_id=%s)", transaction_id)
        result = {
            "status": "charge_cancelled",
            "transaction_id": transaction_id,
            "refund_amount": 8900.00,
            "estimated_credit": "24-48 hours",
        }
        await agent.send_custom_event({"type": "charge_cancelled", "data": result})
        logger.info("💰 Charge cancelled, $%.2f returning — ETA 24-48 hours", result["refund_amount"])
        _log_tool_call("cancel_charge", {"transaction_id": transaction_id}, result)
        tool_cache[cache_key] = result
        return result

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Fraud Detection Agent...")

    async with agent.join(call):
        await agent.say("Thanks for calling fraud support, how can I help you today?")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
