# Adaptive audio interruption

Managed cascade agents can ask an audio classifier to stop the current reply
before a partial transcript is available. Configure a private endpoint implementing
the [gophonic interruption service](https://github.com/GetStream/gophonic/pull/110):

```yaml
interruption:
  endpoint: http://127.0.0.1:8081/v1/interruption
  token_file: /run/secrets/interruption-token
  threshold: 0.5
  timeout: 500ms
```

Equivalent environment variables: `ROUTER_INTERRUPTION_URL`,
`ROUTER_INTERRUPTION_TOKEN_FILE`, `ROUTER_INTERRUPTION_THRESHOLD`, and
`ROUTER_INTERRUPTION_TIMEOUT`. The endpoint is empty by default. For an
unauthenticated loopback service, omit `token_file`.

During audible playback, caller voice activity starts an observation window.
The agent sends the caller's preceding second and observed speech, beginning at
200 ms, at most every 100 ms when no request is in flight. It keeps observing
after low predictions so an acknowledgment can develop into a correction. A
positive prediction uses the existing local playback stop and provider cleanup.
Results are tied to both the reply and the caller utterance. They cannot stop a
newer reply; a 300 ms quiet gap starts a new utterance. Only the current caller's
audio takes this fast path when a caller identity is already established.

This is an additional fast path. Low scores, timeouts, invalid responses and busy
endpoints leave the existing transcript/controller handling active. Requests stop
after two seconds of observed speech or an endpoint error for that utterance.
Native duplex providers retain their own interruption behavior.

Use a threshold measured on development conversations, then test actual playback
stop latency, false stops and missed interruptions. The initial gophonic pilot
does not justify enabling this by default. VAD here is the existing audio-level
detector; an acoustic classifier does not remove its onset errors or echo risk.
