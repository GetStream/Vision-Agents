# Phone & RAG example with Twilio, Stream and Turbopuffer

This example shows you how easy it is to build both inbound and outbound calling agents. 
It uses Twilio for calling, Stream for video edge, and either Gemini or Turbopuffer for RAG. 

Note that for optimal latency you'll want to deploy this sample in US-east. Local development will incur some additional latency due to roundtrips. 

## Step 1 - Setup up your .env

### Clone the repo

```
git clone git@github.com:GetStream/Vision-Agents.git
```

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

### Step 2 - Outbound Call example

The outbound example showcases how you can call a restaurant and use AI to make a reservation. 

A. Start NGROK 

First we'll start [NGROK](https://ngrok.com/) to ensure we have a public address for our local server.
Copy the NGROK url from this tab.

```
ngrok http 8000
```

B. Setup your Twilio phone number

* Login to Twilio
* Phone numbers -> Manager -> Active numbers -> Buy a number
* For the number you've bought set the "A call comes in" webhook to your NGROK URL like this: `https://replaceme.ngrok-free.app/twilio/voice`

C. HTTP & Call

Now create a new tab in your terminal and start the HTTP endpoint

```
cd examples/03_phone_and_rag_example
NGROK_URL=replaceme.ngrok-free.app uv run outbound_phone_example.py --from +1**** --to +1***
```

Remember to set the correct:

* Ngrok domain
* the number you're calling from (as registered in your Twilio account)
* the number you're calling (try with your own cell)

When you run the above command you'll start a HTTP server to handle the Twilio media stream and initiate an outbound call.
That's how easy it is to connect AI to a Twilio phone call.
Next, let's try an inbound example.

### Step 3 - Inbound call

The inbound example is easy since you already have the Twilio number and NGROK up and running.

A. HTTP

Remember the ngrok url from the first tab. And in a new tab start your server

```
cd 03_phone_and_rag_example
RAG_BACKEND=gemini NGROK_URL=replaceme.ngrok-free.app uv run inbound_phone_and_rag_example.py
```

B. Call the number

Call the number you've setup in Twilio and you'll end up talking to the agent. 
The agent will gladly tell you all about Stream's APIs for chat, video and voice.

Note that there is some added latency during development if you're not running in US-east.
For optimal latency run the example in US-east.


## Step 4 - Understanding the examples

### TWIML

Twilio uses a [Twiml](https://www.twilio.com/docs/voice/twiml) syntax to control phone calls. 
We use the [Twilio Stream](https://www.twilio.com/docs/voice/twiml/stream) command to pipe the call to a websocket URL.

### Websocket handling

If you open up inbound_phone_and_rag_example.py you'll see the following endpoint

```python
@app.websocket("/twilio/media/{call_id}/{token}")
async def media_stream(websocket: WebSocket, call_id: str, token: str):
```

The following syntax connects the twilio media stream to a Stream call object

```
await twilio.attach_phone_to_call(stream_call, twilio_stream, phone_user.id)
```

### RAG

RAG can be quite complicated. Have a look at the [RAG docs](https://visionagents.ai/guides/rag) to understand what's possible. 



