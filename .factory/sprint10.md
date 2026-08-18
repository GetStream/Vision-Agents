

## Synthetic LLM testing

In the database store a list of questions to ask the LLM
together with an evaluation prompt for the answer

Support either contains for simple checks.
And evaluations by a simple AI model for more complicated checks

## Synthetic voice testing

Use the same tests as above. But route it through either a pre-recorder or TTS generated full audio pipeline


## Real call evaluations

1. WER tracking
   If a call/agent is being evaluated enable recording in the stream call
   After the call is completed run a slower and more accurate STT model
   Use that to evaluate what the faster STT did during the call

2. After the call completed store a summary

Evaluate the summary against an evaluation script this customer has set up

## Benchmarks

### Latency bench

For each telephony provider we want to simulate calls from UK, California and NYC. And then we want to measure

- The region their data center connects us to for those calls
- The call delay metrics, average, p95 and p99

### Restaurant eval

### Dentist eval

### Salon eval

### 

competitive benchmarks

