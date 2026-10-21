# internal/sandbox/brokertest

The suite every `sandbox.CredentialBroker` passes, so what the session relies on is proved once and checked for each broker. `BrokerContract` (`brokertest.go`) is the suite. It is test code: import it only from `_test.go` files.

## How a broker runs it

One test function builds the suite with `New`, which returns a `Subject` for a session id, the `sandbox.Tokens` the test controls and a logger:

- `Broker` is the broker under test.
- `Held(ref)` is the credential and hosts the provider keeps under a `Mount.Secrets` value.
- `Left()` is everything the broker has left with the provider, of every session.
- `Refuse()` / `Recover()` make the provider refuse every request, then answer again.
- `Live` is true for a broker whose egress asks for the token on every request (egress): a token that cannot be had just now reaches no request.

`daytona/secrets_test.go` (`TestSecretsBrokerContract`) is the example: Daytona's API is an `httptest.Server`; `egress/broker_test.go` (`TestProxyBrokerContract`) runs it against the egress proxy. Tests that need the provider's real wire (egress) run live, per broker (`daytona_live_test.go`).

## What it covers (attack table A1-A13 of wave 13)

| Case | Tests |
| --- | --- |
| A1, A2: the sandbox gets a reference, never the token; the provider keeps it for the grant's hosts only | `TestTheSandboxGetsAReferenceNeverTheToken` |
| A8: two sessions never share a credential | `TestTwoSessionsNeverShareACredential` |
| A9: Close leaves nothing with the provider | `TestClosingLeavesNothingWithTheProvider` |
| A10: a revoked connection is withdrawn; a failed renewal is kept and asked again; a refused withdrawal is an error and is retried | `TestAConnectionRevokedMidSessionIsWithdrawn`, `TestACredentialUnavailableJustNowIsKeptAndAskedForAgain`, `TestAWithdrawalTheProviderRefusesIsAnErrorAndIsRetried` |
| A13: no token in a log line or an error (`TearDownTest`, after every test), on Prepare and on refresh | `TestAProviderThatRefusesLeavesTheTokenInNoError`, `TestARefusedRefreshIsAnErrorWithoutTheToken` |
| A new revision is handed over before the next run; a disconnected or required grant behaves as the session expects | `TestANewRevisionIsHandedOverBeforeTheNextRun`, `TestADisconnectedConnectionGivesNoEnvAndAWarning`, `TestARequiredGrantWithoutACredentialRefusesTheSandbox` |

Spans are not checked: a provider SDK makes its own only when told to, with an exporter of its own, so a span check here would check nothing.
