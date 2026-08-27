# Outbound Call Example

An agent that rings a real phone and holds the conversation when somebody answers.

Stream's SIP is inbound only, so this is not Stream dialling out: a telephony vendor places
the call and bridges the answered leg into a Stream call. The agent joins that call before
the phone starts ringing, so nobody picks up to silence.

## Prerequisites

- Python 3.13 or higher
- A running acceleration router (see [acceleration/README.md](../../acceleration/README.md))
- A telephony vendor configured on it, and a number bought from that vendor
- API keys for:
    - [Stream](https://getstream.io/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents) (for video/audio infrastructure)
    - Whatever providers the router is configured with

Buy a number if you do not have one:

```bash
cd acceleration
go run ./cmd/phone search -country US -area 512
go run ./cmd/phone buy -vendor twilio -number +15125551234
```

## Installation

1. Go to the example's directory

    ```bash
    cd examples/13_outbound_call_example
    ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

3. Create a `.env` file:

   ```
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   STREAM_ACCELERATION_URL=http://localhost:8080
   STREAM_ACCELERATION_CUSTOMER_ID=examples
   OUTBOUND_TO=+15550001111
   ```

## Running the Example

```bash
uv run outbound_call_example.py
```

Your phone rings, and the agent greets you when you answer.

## How it works

The agent is configured once, by name, and named from then on:

```python
await stream.define_agent("john", instructions=INSTRUCTIONS, llm="llm-fast")

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="John", id="agent"),
    llm=stream.Accelerated(config="john"),
    phone=stream.Phone(),
)
```

`stream.Accelerated(config="john")` looks the name up when the agent joins, so the config
can be defined in one place and used in another. Anything else passed to `Accelerated`
overrides what the config says.

`phone=` is where calls are placed. It is a plugin because the phone paths belong to the
acceleration backend, and the core only knows the shape of it.

Then one context manager creates the call, rings the person and joins:

```python
async with agent.outbound_call(from_=held, to=person, call_type="default", call_id="hello"):
    await agent.simple_response("greet the user and let them know you're a friendly AI agent")
    await agent.finish()
```

### What the call can ask for

| Argument | What it does |
| --- | --- |
| `ring_timeout` | Give up after this many seconds rather than reaching voicemail |
| `initial_digits` | Pressed once the person answers, e.g. `ww1234#` for an extension |
| `headers` | Custom SIP headers carried to the person's leg |
| `custom` | Fields put on the Stream call, where the agent can read them |

Vendors do not all support all of these, and one that cannot express a term refuses the
call rather than placing it without. A ring timeout that was silently dropped is a call
sitting in somebody's voicemail for a minute.

Seven of the eight implemented vendors can place a call. DIDWW cannot: it sells numbers and
has no call control API at all, so there is nothing to ask it to dial with. `phone vendors`
says which is which.

## Learn More

- [The acceleration backend](../../acceleration/README.md)
- [The stream plugin](../../plugins/stream/README.md)
- [The accelerated agent example](../00_accelerated_example)
- [Main Vision Agents README](../../README.md)
