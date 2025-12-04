# Phone & RAG example

This example teaches you how to handle inbound or outbound phone connects and RAG.
It uses Twilio for the voice connection and either Gemini or turbopuffer for RAG.
Note that this example will have relatively high latency unless you run it in US-east/ close to the twilio/stream servers

## Requirements to run the example

### Set your .env file

Create a .env file and set the following vars

```
STREAM_API_KEY=
STREAM_API_SECRET=
GOOGLE_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TURBO_PUFFER_KEY=
```

### Running the Outbound Call example

A. Start NGROK 

```
ngrok http 8000
```

B. HTTP & Call

Copy the ngrok url from the first tab. And in a new tab start your http endpoint

```
cd 03_phone_and_rag_example
uv sync
NGROK_URL=replaceme.ngrok-free.app uv run outbound_phone_example.py --from +1**** --to +1***
```

This will start an HTTP server that can accept the twilio media stream. It also initiates the call


### Running the example - Inbound call

The inbound call example is more complex. It showcases RAG and inbound call handling

A. Start NGROK 

```
ngrok http 8000
```

B. HTTP

Copy the ngrok url from the first tab. And in a new tab start your http endpoint

```
cd 03_phone_and_rag_example
uv sync
RAG_BACKEND=turbopuffer NGROK_URL=replaceme.ngrok-free.app uv run phone_and_rag_example.py
```

C. Twilio console

For one of your phone numbers set the webhook url to

```
replaceme.ngrok-free.app/twilio/voice
```

D. Call the number

Call the number. You'll end up talking to the agent


### Running the example - Outbound call

```
cd 03_phone_and_rag_example
uv sync
RAG_BACKEND=turbopuffer TWILIO_NUMBER=numberhere uv run outbound_example.py
```


## Understanding the examples

### Twilio

Twilio works by creating a websocket based media stream.
This logic is the same for both inbound and outbound. 



## RAG

RAG is a relatively complicated topic. Have a look at the full docs on



