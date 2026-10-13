# Telephony latency

Measures mouth-to-ear latency between phone numbers on different carriers in the ten most
populous US cities, and splits it into what the calling carrier adds and what the called carrier
adds. What and why: [the spec](../../../.factory/evals/telephony-latency.md).

## How a call is measured

The bench is a SIP endpoint on a public address. Every number is pointed at it, so when number A
calls number B:

1. The bench asks A's vendor to dial B, bridging the answered call to `sip:probe-<id>@bench`.
2. B's vendor delivers the call to `sip:<B>@bench`, and the bench answers it.
3. A's vendor sees the answer and bridges its side to the bench as well.

Both legs of the call now end on the bench, one per carrier. The bench sends a timestamp as a
360 ms burst of tones on one leg (a sync pattern, the millisecond four bits per tone, a CRC-8),
and decodes it off the other. The latency is when the burst's first sample arrived minus the time
written in it. Stamps go both ways, five each way per call by default.

The figure includes the bench's own trip to each carrier's edge, so run it from the regions your
agents run in, and compare carriers within a region rather than across regions. Arrival is taken
off the network, before any jitter buffer, so a handset would add its own on top.

## Running it

Credentials come from the environment: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
`TELNYX_API_KEY`, `TELNYX_CONNECTION_ID` and `TELNYX_CALL_CONTROL_ID`. Optionally `IPINFO_TOKEN`
for the report's address lookups.

```bash
cd acceleration
go run ./evals/telephony-latency provision            # buys 2 numbers per vendor per city: 40 numbers
```

Then on a VM in each GCP region, with UDP 5060 and 20000-20999 open to the carriers:

```bash
go run ./evals/telephony-latency run -host <vm public ip> -region us-east1
```

Each run points every number at that VM, places every call in every city 10 times, and appends
one line per call to `results-<region>.jsonl`. Regions take turns, since a number can point at one
bench at a time. Then:

```bash
go run ./evals/telephony-latency report -out report.md results-*.jsonl
go run ./evals/telephony-latency release              # stops the monthly charge
```

Useful flags on `run`: `-cities Chicago,Houston` and `-vendors twilio` to narrow it, `-runs 1`
for a smoke test, `-scope all` to call between cities as well, `-parallel` for calls at once.

### Carrier setup

- **Twilio** needs nothing: the bench sets each number's voice URL to TwiML that dials it.
- **Telnyx** routes inbound calls through `TELNYX_CONNECTION_ID`. Make it an FQDN/IP connection
  pointing at the bench's address, with the destination number in E.164. Each region's bench
  needs its own connection, or the connection repointed before each run. This is not the
  router's `TELNYX_CONNECTION_ID`, which is a TeXML application.
- **Telnyx** places calls through the Call Control Application `TELNYX_CALL_CONTROL_ID`, in three
  requests: dial the bench, dial the callee linked to that call, bridge the two. The phone
  package's Telnyx `Dial` is not used, because it sends the bridge address as `link_to`, which
  Telnyx reads as a call control id, so nothing would be bridged.

## The report

For each region: a vendor × vendor matrix of all calls and of each city, in both directions, with
the median, p95 and number of calls in each cell; the addresses each vendor's media came from,
located by the clouds' published ranges (AWS, GCP, Oracle) or the network's name and ipinfo.io;
ring and bridge times; and failures.

**What each vendor adds** fits every cell as `baseline + caller share + callee share`. Only the
differences between vendors in the same role can be told apart, so each role's shares average to
zero and the rest stays in the baseline: "twilio as caller +12 ms" means calls Twilio places are 12
ms slower than the average caller's. A large "unexplained" column means a pair of vendors is slow
together in a way neither is alone, such as one bad interconnect. Two numbers per vendor per city
are bought so a vendor calling itself is a cell; without that diagonal the split has no answer.
