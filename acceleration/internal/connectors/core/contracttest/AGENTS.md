# internal/connectors/core/contracttest

The suites every adapter of one kind runs, so what the core relies on is proved once and checked for each adapter. `SchemeContract` (`scheme.go`) is the one for `core.Scheme`, `SourceContract` (`source.go`) the one for `core.ToolSource`. It is test code: import it only from `_test.go` files.

## What `SourceContract` proves

The subject runs a provider offering three tools: `ToolEcho` (`{"text": string}`, answers the text), `ToolLarge` (answers more than `core.MaxResultBytes` of two-byte characters) and `ToolBroken` (reports an error whose only content is an image). `ChangeSchema` gives `ToolEcho` another input schema. The contract learns the name each tool is offered under from a toolset granting it alone, so it assumes nothing about how a source names tools.

| Test | What holds for every tool source |
| --- | --- |
| `TestDiscoverGivesTheSameDigestForTheSameSchema`, `TestAChangedSchemaHasAnotherDigest` | A digest is stable for one schema and moves with it |
| `TestOnlyTheGrantedToolsAreOffered`, `TestAnUngrantedToolIsNeverDispatched` | `Open` offers exactly the granted tools; a call to another never reaches the provider |
| `TestAToolWhoseSchemaChangedSinceItWasGrantedIsHidden` | A grant pins the schema: after a change the tool is neither offered nor called |
| `TestArgumentsTheSchemaRefusesNeverReachTheProvider`, `TestAGrantedToolAnswers` | Arguments are checked against the schema before anything is sent |
| `TestAResultOverTheCapIsCutWithTheMarker` | A long result fits `core.MaxResultBytes`, ends with `core.TruncatedMarker` and is valid UTF-8 |
| `TestAnErrorWithNoTextIsStillAnError` | A tool's own failure is a `*core.ToolError`, even with no text |

A new source adopts it with one test function, as `sources/mcp/source_test.go` (`TestMCPSourceContract`) does.

## What `SchemeContract` proves

| Test | What holds for every scheme |
| --- | --- |
| `TestARoundTripPutsTheCredentialOnTheRequest` | Begin, the browser if there is one, Complete, Retrieve, Wrap: the provider takes the request, and refuses it without Wrap. A credential that is not due comes back as stored. It has an expiry exactly when it can be due. An anonymous scheme holds no secret and changes nothing on the request |
| `TestConcurrentResolvesCommitAtMostOneNewRevision` | Eight resolves at once through a locked in-memory `core.CredentialStore`: no new revision for a credential that is not due, exactly one for one that is, and one renewal at the provider |
| `TestConcurrentRetrievesOfACredentialThatIsNotDueAllHandItOutAsStored` | Retrieve without a lock is safe to call at once (run with `-race`) and changes nothing |
| `TestStoredCredentialsItCannotReadAreRefusedWithoutQuotingThem` | Another scheme's, a newer version's or a cut-off payload: Retrieve and Revoke refuse it |
| `TestAnErrorAboutAnUnusableValueDoesNotQuoteIt` | Each supplied value with CR LF after it: whether `Complete` refuses it or not, no error quotes it. For an interactive scheme, a callback made of the secrets is refused |
| `TestWrapAppliesOnlyACredentialThisSchemeIssued` | A credential under another scheme's name puts no secret on the wire |
| `TestClassifyMakes…` (six) | One answer or more per outcome, from a real local server: 200 OK; 401 `invalid_token` InvalidGrant; 403 `insufficient_scope` ScopeRequired with its scopes; 429 RateLimited with `Retry-After`; 503 and a refused dial Transient; 500 and a connection closed after the request Uncertain |
| `TestRevokeSaysWhatItDid` | nil means the provider refuses the credential afterwards; an error (`Subject.RevokeErr`) means it still takes it, and is not an `*core.OutcomeError` |

After every test, `TearDownTest` looks for every secret `Subject.Secrets` named in every error text, every URL Wrap sent or Begin returned, and every line written to the test's logger, which is also `slog.Default()` while the test runs (JSON and text handlers both). It looks for each secret in every form `forms` lists: as it is, percent-encoded as `net/url` writes a query value, a path segment, a path, a fragment and userinfo, and escaped as both slog handlers write a string. A hand-rolled encoding (lowercase hex, base64) is not among them.

A fixture secret should hold characters those forms write differently (`+ / = ; , @ : ? # " \` and an inner space, as `apikey_test.go`'s key does), so a secret in a URL or a log line cannot pass because it was written encoded.

## Adopting it

A new scheme adds one test function in its own package, beside its own suite:

```go
func TestBasicSchemeContract(t *testing.T) {
	suite.Run(t, &contracttest.SchemeContract{New: func(t *testing.T, _ *slog.Logger) contracttest.Subject {
		p := provider(t) // an httptest server that answers 2xx only with the credential on the request
		return contracttest.Subject{Scheme: basic.New(), Supplied: supplied, Transport: p.Client().Transport, Call: call(p), Secrets: secrets, RevokeErr: basic.ErrNotRevocable}
	}})
}
```

`supplied` is what `Complete` gets, `call(p)` builds a request to `p`, and `secrets` returns the supplied password.

An interactive scheme also sets `Consent` (the browser: `fakeprovider.Server.Consent`), `RedirectURI` and `Manifest`; a renewing one sets `Expire` and `Renewals`, and gives the scheme a clock that is safe to read from several goroutines. `schemes/oauth2code/contract_test.go` is the example.

## Rules

- **No mocks.** Every answer the suite reads comes from a real local server (`httptest`), and the provider a subject names is one too (`.claude/skills/go-testing/SKILL.md`). Check: `grep -rn -i --include='*.go' 'mock' internal/connectors/core/contracttest` prints nothing.
- **It imports `core` and nothing else under `internal/connectors/`**, so it sits inside core's guard (`core/guard_test.go` walks this directory too) and any scheme can import it. Check: `go list -f '{{join .Imports "\n"}}' ./internal/connectors/core/contracttest | grep internal/connectors/` prints only `.../core`.
- **What it asserts holds for every scheme.** A rule only some schemes keep belongs in that scheme's own suite, or behind a `Subject` field that says which kind of scheme it is (`Anonymous`, `Consent`, `Expire`). Check: `go test -race ./internal/connectors/schemes/...` runs it for all four.
- **Every hardcoded value says where it comes from**, as the comments on the Classify tests and on `concurrency` do.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/...`. The package has no tests of its own; each scheme's `Test*SchemeContract` runs it, and a contract change is checked by breaking a scheme by hand and watching the suite fail (the PR that added it lists the mutations it was run against).
