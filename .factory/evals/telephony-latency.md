# Latency benchmark

How much delay does a phone call add, and whose fault is it: the carrier placing the call or the
one receiving it. The answer decides which vendor an agent's number should live on in each city,
and it is not something a vendor's own dashboard will tell you.

It lives in `acceleration/evals/telephony-latency` and is written in Go, using the packages in
`acceleration` for buying numbers, pointing them somewhere and placing calls
(`internal/phone`, its vendors, and `sipbridge` for reading SDP).

## Cities

The ten most populous cities in the USA are the measurement points:

New York
Los Angeles
Chicago
Houston
Phoenix
Philadelphia
San Antonio
San Diego
Dallas
Jacksonville

## Providers

Use Twilio and Telnyx for now. We will expand later

## Benchmark/Eval

Register phone numbers in each city on each provider. Every number both starts an outgoing call
and answers an incoming one, so each vendor is measured in both roles.

Build a matrix of each provider calling each other provider in a city: have each phone number
call each other phone number. Two numbers per vendor per city, so a vendor calling itself is a
cell too; without that diagonal the matrix cannot separate the caller's share from the callee's.
Calls between cities are a wider scope the bench can also run.

For each cell in the matrix run the test 10 times, which is 10 separate calls.

When phone number A calls phone number B, track the following
- IP used for provider A, for signalling and for media
- IP used for provider B, for signalling and for media
- Lookup the location of those 2 IPs
- See if they match IPs used by any major cloud providers
- Measure end to end latency by sharing synthetic audio on the call

### Measuring the latency

Encode a timestamp as an audio fragment, so you can measure the delay. The fragment is a burst of
tones, one per four bits, carrying the millisecond it was sent and a checksum. The other end
decodes it and the latency is the time it was heard minus the time written in it. Tones survive
G.711 and a carrier's transcoding where a waveform would not, and because the time travels in the
audio no table of what was sent when is needed to read it.

Both legs of a call end on the bench, so each call measures both directions: A's side to B's,
and B's side back to A's.

## Latency report

Which IPs/data centers were used by the various vendors in each cell of the report/matrix.
In each cell show the end to end latency (median, p95, how many calls).

Below that show a report about how much latency each vendor adds.
This matrix allows you to calculate how much latency is added by the caller and by who you're
calling: fit every cell as a baseline plus what the calling vendor adds plus what the called
vendor adds, and report each vendor's share in both roles.

## Hosting the bench

Use SIP, Run a little SIP enabled go program on each GCP region. Each region's run writes its
own results, and the report puts the regions side by side, because a vendor that is fast from
us-east1 can be slow from us-west1.
