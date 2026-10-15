# Go security review

The router in `acceleration/`. Read [AGENTS.md](../../../AGENTS.md#http-api) for how
operations are declared and the `go-testing` skill for how suites are written, then hold
the code to the rules below. Each rule names the code that enforces it and the test that
proves it; a rule with no test is a finding even when the code is right.

How a caller is classified lives in `internal/auth/auth.go` (`keyed.Authenticate`), and
what the API does with that in `internal/api/server.go` (`ServerSideFrom`, `CallerFrom`,
`KindFrom`, `OwnerFrom`). The model, in short:

- **Server-side** is a backend: `Stream-Auth-Type: server` *and* a token with `server: true`
  and no `user_id`. It may name the user it acts for in `X-Stream-User-Id`.
- **Client-side** is an end user's device: a token naming a `user_id` (authenticated or
  guest), or one naming nobody (anonymous). It reaches only operations marked
  `x-client-accessible`, and only what it owns.
- **Public** is an operation declaring `Security: []`: health, and hooks that prove their
  sender with a signature instead.

## Coverage

Run these before reading code. They are what makes a missed endpoint fail loudly.

```bash
cd acceleration
go test -tags integration ./internal/api -run 'TestPostureSuite|TestSessionOwnershipSuite' -count=1
go test ./internal/auth -count=1
```

`PostureSuite` (`internal/api/posture_test.go`) walks every operation in the spec, so it
only protects routes the spec knows about. Check:

- Every route is declared: with `huma.Register`, or in `internal/api/handwritten.go` for a
  socket or a stream. A handler mounted straight on the chi router or a `ServeMux` is
  invisible to the middleware and to the posture test. Grep a diff for
  `\.(Get|Post|Put|Patch|Delete|Handle|HandleFunc|Mount)\(` outside `huma.Register`.
- `TestTheCommittedSpecIsRenderedFromTheOperations` passes, so `api/openapi.yaml` is what
  the server enforces.

## Rules

### 1. Backend endpoints only work from a backend

Every operation not marked `x-client-accessible` answers a client-side caller 403.

- Enforced by the middleware reading the mark off the spec (`clientAccessibleExtension` in
  `server.go`), not by a list. Tested by
  `TestEveryOperationTheSpecDoesNotOpenIsRefusedToAUsersDevice`.
- The test must call with each client kind, not only one: an authenticated user, a guest,
  an anonymous caller, and a server token presented with `Stream-Auth-Type: jwt` or no
  header at all. If the suite walks the spec with one client only, that is a gap to report.
- `Stream-Auth-Type: server` with a user token, and a server token in a socket's query
  string, are both client-side. Tested by `TestAPIKey` and
  `TestAServerTokenInASocketsQueryStringIsNotABackend`.
- A new `x-client-accessible` mark is a decision, not a default. The PR has to say why an
  end user's device needs it, and come with ownership tests (rule 3). Flag any mark on an
  operation that writes configuration, keys, policies, agents, phone numbers, connectors
  or billing.
- A handler that branches on `ServerSideFrom(ctx)` to do more for a backend must have a test
  that the client-side branch does less.

### 2. Client endpoints need an API key and a token

In `api_key` mode a client-side operation is reached with both the `X-Api-Key` and a valid
token signed with that key's secret, or with neither and a 401.

- Missing key, missing token, unknown key, malformed key, wrong secret, expired token, no
  `exp`, `alg: none`, and an algorithm other than HS256 are all a 401, and the same 401 body
  (`assertRefusedAlike`), so a caller cannot learn which keys exist.
- Tested in `internal/auth/auth_test.go` (`TestAPIKey`) for the authenticator and in
  `PostureSuite` for the wire. `PostureSuite` checks the no-credential case on one path;
  report as a gap any client-accessible operation, socket included, with no test that it
  answers 401 without a credential. The fix is a spec-walking test like the 403 one.
- Sockets take `api_key` and `token` as query parameters. Check that the socket
  authenticates before upgrading and that a failed check never upgrades.
- `noauth` and `proxy` trust headers. Check that nothing makes them the default and that a
  deployment cannot fall into `noauth` by leaving a variable unset (`auth.ParseMode`).
- An app turning away a level (anonymous, guest) is a 403 at the door, for every operation.

### 3. A client cannot pose as another user

Who a client-side caller is comes from its verified token and nowhere else: never from the
body, a query parameter, a path segment or `X-Stream-User-Id`.

- In handlers, the user is `CallerFrom(ctx).UserID` or `OwnerFrom(ctx)`. Flag any
  client-accessible handler that reads a `user_id`, `user`, `sender`, `owner`, `from`,
  `author`, `as`, `on_behalf_of` or similar field from the request and acts on it. Grep the
  request structs:

  ```bash
  rg -n 'json:"(user|user_id|users|owner|owner_id|sender|sender_id|from|author|as_user|on_behalf_of|customer_id|app_id|organization_id)[",]' acceleration/internal/api
  ```

  Each hit on a client-accessible operation either applies only when `ServerSideFrom(ctx)`
  (like `user_id` in `session_query.go`, which narrows an end user to their own) or is a
  finding.
- Anything a client writes into a conversation — a message, a response, a tool result, an
  event — is attributed to the principal, not to a name in the payload. Test: a device sends
  a message naming another user, and the stored and broadcast message names the device's own
  user.
- `X-Stream-User-Id` is believed only for a server-side caller. For an anonymous caller it is
  a name kept apart from every verified user by its kind. Tested by
  `TestClaimingAUsersNameWithoutATokenReachesNothing` and
  `TestAGuestIsNotTheSignedInUserOfTheSameName`.
- Ownership is checked on every read, write, list and socket of a resource a client can
  name by id, and a resource that is not theirs is a 404, not a 403
  (`TestOneUsersConversationIsOutOfAnothersReach`). Lists filter by owner in the query, not
  after it.
- The same user id in another app reaches nothing
  (`TestAnotherAppsUserReachesNothingEvenByTheSameName`).

A new client-accessible operation that touches a resource needs, at least: the owner
reaches it, another user gets a 404, an anonymous caller naming the owner gets a 404, and
another app's user of the same name gets a 404. Copy `SessionOwnershipSuite`.

### 4. Tenants stay apart

- Every store query that reads or writes tenant data is scoped by customer, app or
  organization id taken from the context, never from the request. Flag a `store` method
  called with an id from the body, or one whose `WHERE` has no tenant column.
- A path id from one app looked up with another app's credential is a 404.
- Caches (`appconfig`, Redis keys, relay channels) are keyed by tenant. Flag a key built
  from a resource id alone.

### 5. Public endpoints prove their sender

An operation with `Security: []` (hooks, Telnyx, provider callbacks, OAuth returns) has no
caller to classify, so it carries its own proof.

- The signature is checked over the raw body, before anything is parsed or acted on, with
  `hmac.Equal` or `subtle.ConstantTimeCompare`, never `==` or `bytes.Equal`.
- With no secret configured the endpoint refuses everything (`HookSecret`,
  `TelnyxPublicKey`), rather than accepting unsigned requests.
- A timestamped scheme rejects old timestamps, so a captured request cannot be replayed.
- The body is read through `http.MaxBytesReader` (see `hookbody.go`).
- OAuth state, handoff tokens and cookies are compared in constant time and used once
  (`authorizations.go`).
- Each has a test that an unsigned, wrongly signed and stale request is refused.

### 6. Outbound requests cannot be aimed inside

Any URL a caller supplies — knowledge URLs, MCP and connector endpoints, event
destinations, webhooks, callback URLs, images to fetch — goes through
`egress.ValidatePublicHTTPSURL` and an `egress.NewClient`, which refuse private, loopback,
link-local and metadata addresses (`169.254.169.254`) at dial time, so a DNS rebind or a
redirect cannot get round a check made earlier. Flag `http.Get`, `http.DefaultClient` or a
fresh `http.Client` on a caller's URL.

### 7. Secrets stay secret

- Client secrets, API secrets, OAuth tokens and provider keys are sealed with the KEK at
  rest (`auth.Sealer`), returned once at creation at most, `readOnly`/write-only in the
  schema, and never in a list or get response afterwards.
- Never logged: check new `slog`/log lines and error messages for tokens, keys, `Authorization`
  headers, query strings carrying `token=`, and request bodies.
- A 500 says only "something went wrong" (`apierror.go`). Flag an error that returns a
  wrapped driver, provider or upstream message to the caller.
- The ops key (`X-Ops-Key`) is compared in constant time and turns the paths off when empty.

### 8. Input that reaches SQL, the shell or the filesystem

- SQL uses placeholders. `fmt.Sprintf` into a query is allowed only for identifiers from a
  fixed list in code (as in `store/datamove.go`), never for a value from a request.
- Sandbox commands, skill files and anything written to disk: no caller-supplied path is
  joined without rejecting `..`, absolute paths and symlinks out of the root; no caller
  string is passed to `sh -c`.
- Templates rendering caller text into HTML use `html/template`.

### 9. Abuse limits

- Client-side operations that spend money (responses, tokens, calls, TTS) count against the
  daily limits (`ROUTER_RATE_LIMIT_*`) and answer 429 with `Retry-After`. A new one that
  spends and is not counted is a finding.
- Request bodies are bounded (Huma's `MaxBodyBytes`, or `MaxBytesReader` on hand-served
  routes). Sockets bound frame size and close idle connections.
- List endpoints cap `limit` (see the `pagination` skill).
- Outbound calls carry a context with a timeout.
- A long-lived stream (an STT or TTS websocket, a realtime session) bounds its handshake,
  not the whole call: `websocket.Dialer{HandshakeTimeout: ...}`, as in
  `internal/stt/cartesia/cartesia.go`. A fixed deadline would end the call mid-conversation.

### 10. Browser surface

- Any origin is allowed by design outside `noauth` (`originAllowed` in `sessionws.go`),
  because the credential is a key and a token a page has to hold, never a cookie. So the
  rules are: no `Access-Control-Allow-Credentials`, and no cookie that authenticates an API
  call. `noauth` answers loopback origins only. Tested in `cors_test.go`.
- Sockets that a browser opens check the origin the same way.

### 11. Dependencies

```bash
cd acceleration
go run golang.org/x/vuln/cmd/govulncheck@latest ./...
```

Report a vulnerability only if the vulnerable symbol is reachable (govulncheck says so).
New dependencies in `go.mod` are a question for the PR: who maintains it, and does the
standard library already do it.
