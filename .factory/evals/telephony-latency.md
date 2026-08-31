# Latency benchmark

Create an evals folder and inside of that a telephony-latency folder
Use Go to build this test. Use the packages in acceleration as relevant (for buying numbers, calling etc.)

## Cities

These are the top cities we want to measure latency for

New york
San Francisco
Seattle
Los Angeles
Chicago
Houston
Phoenix
Austin
Philadelphia
San Antonio
San Diego
Dallas
Fort Worth
Denver
Boston
Miami

## Providers

Use Twilio and Telnyx for now. We will expand later

## Benchmark/Eval

Register a phone number in each city on each provider.
In a matrix, have each phone number call each other phone number.

When phone number A calls phone number B, track the following
- IP used for provider A
- IP used for provider B
- Lookup the location of those 2 IPs
- See if they match IPs used by any major cloud providers
- Measure end to end latency by sharing some synthetic audio on the call

## Latency report

Which IPS/Data centers where used by the various vendors in each cell of the report/matrix
In each cell show the end to end latency

Below that show a report about how much latency each vendor adds.
This matrix allows you to calculate how much latency is added by the caller and by who you’re calling

## Hosting the bench

Use SIP, Run a little SIP enabled go program on each GCP region. 
