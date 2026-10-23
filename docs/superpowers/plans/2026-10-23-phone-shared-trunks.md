# Phone: shared trunks for campaigns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Outbound calls stop costing Stream SIP trunks. A customer's numbers share one Stream trunk per vendor, up to 50 numbers on each, so parallel calls from one number work (AI-1279, AI-894), trunks stop leaking toward Stream's cap of 25 (AI-1318), and every inbound caller still gets a call of their own (AI-988).

**Architecture:** A number gets a *line*: a place on its group's shared trunk (Stream app + customer + vendor) and a routing rule of its own whose call id is `{{#if x_stream_call_id}}{{x_stream_call_id}}{{else}}{{uuid}}{{/if}}`. A provider that can carry the call id on its leg to Stream (Twilio, Telnyx, `sip_trunk`) dials through the line with the session id in `x-stream-call-id`. Every other provider keeps today's trunk per call. A placed call still records a `call_resources` row, marked shared, so call sites, sandbox minutes and the hook's "the router placed this" check keep working.

**Tech Stack:** Go 1.27 (`acceleration/`), bun + Postgres, goose migrations, `getstream-go/v5 v5.2.0`, testify suites, `chattest.Server` as the fake Stream.

**Spec:** [Phone: shared trunks for campaigns](https://claude.ai/artifact/DJxp4PStvu5aUiuv9rJ6GR) (Claude Doc). Background for the move to chat: [Moving sipbridge into chat: outbound SIP](https://claude.ai/artifact/7MA6UNwxMfBag8rMtt32GX).

**Branch:** a new branch from `accelerate`, e.g. `alialiyev/ai-1279-shared-trunks`.

## Global Constraints

- Read `.claude/skills/go/SKILL.md` and `.claude/skills/go-testing/SKILL.md` before writing Go. Testify suites, no mocks, one test per behaviour named as the sentence it proves, `//go:build integration` for anything on Postgres.
- Errors: `stack.Wrap` where they enter, `phone: <what was being done>: %w`, never log and return.
- A trunk holds at most 50 numbers (chat `payload/create_sip_trunk.go:35`, `update_sip_trunk.go:11`). An app holds at most 25 trunks (chat `create_sip_trunk.go:61`).
- Groups never mix customers or vendors. `sip_trunk` is one more vendor.
- The call id is always the session id. The header is `x-stream-call-id`; Stream exposes it to the template as `x_stream_call_id` (chat `call_request.go:235`).
- Header carriers: Twilio and Telnyx append `?x-stream-call-id=<id>` to the `<Sip>` URI; `sipbridge` adds `X-Stream-Call-Id: <id>` to its INVITE. Never through `Outbound.Headers`, which every vendor but didww refuses.
- Vonage, Plivo, Bird, Sinch and Bandwidth keep today's per-call path unchanged.
- Nothing new goes on the router's `sip_trunks` table (the chat move takes it over).
- No change to the public API shape. `stream_trunk_id` in a number's response keeps meaning "attached for inbound".
- Never `git stash` bare in this worktree; commit WIP instead.

## Review Focus

1. **No KEK (`ROUTER_AUTH_KEK` unset, `s.sealer == nil`).** A shared trunk's password must be stored to be reused, so without a key the call falls back to a trunk of its own and still places. Test in Task 5.
2. **Stream fails halfway through joining a number to a trunk** (trunk updated, rule refused). The number must not be stored on a trunk whose rule does not exist, and the trunk's list must not keep it. Test in Task 4.
3. **The deployment's own app** (`StreamAppPK` stored as null, read back as 0). Its numbers group like any app's. Test in Task 4.
4. **A full trunk loses a number, then a new number arrives.** The new one fills the free place instead of opening a sibling. Test in Task 6.
5. **A dial that fails after the call resource was recorded.** No row is left behind to block or bill anything. Test in Task 5.

---

### Task 1: Live checks that can stop the design

Throwaway: nothing from this task is committed except the results written into the spec doc. Run with the `phonecheck` skill (`.claude/skills/phonecheck/SKILL.md`) on a test app, as the 2026-10-23 header test did: a spike in `Service.Call` only.

Already settled, no need to test: Twilio and Telnyx carry the header (live, 2026-10-23); `UpdateSIPTrunk` keeps the password when none is sent (chat `update_sip_trunk.go:69`); chat has no cap on routing rules.

- [ ] **Step 1: Spike.** In `Service.Call`, make the per-call rule's call id the full template `{{#if x_stream_call_id}}{{x_stream_call_id}}{{else}}{{uuid}}{{/if}}`, append `?x-stream-call-id=<callID>` to `outbound.Bridge.URI`, and before `provider.Dial` call Stream `GetOrCreateCall(ctx, "agent", callID, &GetOrCreateCallRequest{Data: &CallRequest{CreatedByID: <a user id>, Custom: {"spike": "1"}}})`.
- [ ] **Step 2: Check A, the fallback.** Attach a Twilio number on the spike router and call it from a person number. Expected: the call reaches a worker in a new call (the `{{else}}` branch rendered). Then place an outbound call. Expected: `outcome=ok`, the leg in `agent:<session id>`.
- [ ] **Step 3: Check B, parallel legs.** Two `phonecheck outbound` runs at once from the same router number to two person numbers, both against one trunk (spike: reuse the first call's trunk for the second instead of creating one). Expected: both `outcome=ok`.
- [ ] **Step 4: Check C, the call's custom.** From the Step 2 outbound run, read the call (`GetCall`). Expected: `custom.spike == "1"`, and record whether `phone_number` from the rule's custom data was added. Record which `created_by_id` Stream accepted.
- [ ] **Step 5: Write the results** into the spec doc's "Open questions and live checks". If A or B fails, stop and bring it to the human; the rest of this plan assumes both pass.
- [ ] **Step 6: Clean up** as the phonecheck skill says, and `git checkout -- acceleration/internal/phone/service.go`.

### Task 2: Stream client: update a trunk, update a rule, create a call

**Files:**
- Modify: `acceleration/internal/phone/stream.go`
- Modify: `acceleration/internal/conversation/chattest/chattest.go` (handlers near 660-700)
- Test: `acceleration/internal/phone/stream_test.go` (`PhoneSuite`)

**Interfaces:**
- Produces:
  - `func (s *Stream) UpdateTrunk(ctx context.Context, id string, trunk Trunk) error`: sends `Name`, `Numbers`, `AllowedIps` (none has omitempty in the SDK, so all three are always resent); `Password` only when `trunk.Password != ""`.
  - `func (s *Stream) UpdateRoute(ctx context.Context, id string, route Route) error`: the same request as `CreateRoute` builds, via `UpdateSIPInboundRoutingRule`. Factor the request into `routeRequest(route Route)` shared by both.
  - `func (s *Stream) CreateCall(ctx context.Context, callType, callID string, custom map[string]string) error`: `GetOrCreateCall` with `Data.Custom` and `Data.CreatedByID` = the id Task 1 Step 4 recorded, as a const `callCreator`.
  - `chattest.Server`: `PUT /video/sip/inbound_trunks/{id}` and `PUT /video/sip/inbound_routing_rules/{id}` that replace what `Trunk(id)`/`Rule(id)` return; `POST /video/call/{type}/{id}` recorded, read with `func (f *Server) Call(callType, id string) map[string]any`; a new opt-in `func (f *Server) CapTrunks(n int)` refusing the n+1th trunk with `"your application reached the maximum number of SIP trunks (25)"`; and a refusal of more than 50 numbers on a trunk (create and update), always on. `OneTrunkPerNumber()` also applies on update, ignoring the trunk being updated.
  - `func (f *Server) TrunkNumbers(id string) []string`.

- [ ] **Step 1: Write the failing tests** in `stream_test.go`:
  - `TestUpdatingATrunkKeepsItsPasswordWhenNoneIsGiven`: create a trunk, `UpdateTrunk` with two numbers and no password; `fake.TrunkNumbers(id)` equals both, `fake.TrunkAuth(id)` password unchanged.
  - `TestATrunkTakesNoMoreThanFiftyNumbers`: `UpdateTrunk` with 51 numbers returns an error.
  - `TestUpdatingARouteChangesItsCallID`: `UpdateRoute` with `CallID: "x"`; `fake.Rule(id)["direct_routing_configs"]["call_id"] == "x"`.
  - `TestACreatedCallCarriesItsCustom`: `CreateCall(ctx, "agent", "abc", {"k":"v"})`; `fake.Call("agent","abc")["custom"]["k"] == "v"`.
- [ ] **Step 2: Run, expect FAIL** (undefined methods): `cd acceleration && go test ./internal/phone -run TestPhone`
- [ ] **Step 3: Implement** the three methods and the fake's handlers.
- [ ] **Step 4: Run, expect PASS**: `go test ./internal/phone ./internal/conversation/chattest`
- [ ] **Step 5: Commit** `feat(phone): update trunks and rules, create calls at Stream`

### Task 3: The call id on the leg to Stream

**Files:**
- Modify: `acceleration/internal/phone/phone.go`, `twilio/twilio.go`, `telnyx/telnyx.go`, `telnyx/texml.go`, `siptrunk/siptrunk.go`, `sipbridge/config.go`, `sipbridge/dial.go`, `sipbridge/sipleg.go`
- Test: `twilio/twilio_test.go`, `telnyx/telnyx_test.go`, `siptrunk/siptrunk_test.go`, `sipbridge/sipleg_test.go`

**Interfaces:**
- Produces (in `phone.go`):
  - `const CallIDHeader = "X-Stream-Call-Id"`.
  - `type StreamLeg int` with `StreamLegPerCall StreamLeg = iota` (a trunk and rule made for the call) and `StreamLegShared` (the number's line, the call id on the leg). The chat move adds a third value later.
  - `type StreamLegger interface { StreamLeg() StreamLeg }`, an optional interface like `DigitsPlayer`. `func StreamLegOf(p Provider) StreamLeg` returns `StreamLegPerCall` for a provider that does not implement it.
  - `Outbound.CallID string`: set only on the shared path; empty means no header.
  - `func (b Bridge) WithCallID(id string) Bridge`: appends `?x-stream-call-id=<url.QueryEscape(id)>` to `URI`; empty id returns `b` unchanged.
- `sipbridge.CallParams.CallID string`; the Stream leg's INVITE carries `CallIDHeader` when it is set. Only the Stream leg, never the customer leg.
- Twilio, Telnyx and `siptrunk.Provider` implement `StreamLeg() phone.StreamLeg { return phone.StreamLegShared }`.

- [ ] **Step 1: Write the failing tests:**
  - Twilio `TestTheTrunkLegCarriesTheCallID`: `Dial` with `Outbound.CallID: "abc"`; the posted TwiML's `<Sip>` URI ends in `?x-stream-call-id=abc`. `TestAnOutboundWithoutACallIDDialsThePlainURI`.
  - Telnyx `TestTheTrunkLegCarriesTheCallID` on the TeXML `<Sip>` URI, same assertions; inbound `RenderInbound` output unchanged (`TestAnInboundCallIsSentToTheNumbersTrunk` still passes).
  - sipbridge `TestTheStreamInviteCarriesTheCallID`: the INVITE the Stream leg sends has header `X-Stream-Call-Id: abc`; `TestTheCustomerInviteCarriesNoCallID`.
  - siptrunk `TestTheBridgeIsGivenTheCallID`: `Dial` with `CallID` puts it in `sipbridge.Config.Call.CallID`.
  - phone `TestAProviderWithoutAStreamLegIsPerCall`: `StreamLegOf` of the `stub` vendor is `StreamLegPerCall`.
- [ ] **Step 2: Run, expect FAIL**: `go test ./internal/phone/...`
- [ ] **Step 3: Implement.** Twilio's `dialBridge` and Telnyx's `renderTexml` take the bridge already passed through `WithCallID` in `Dial`, so inbound paths stay untouched.
- [ ] **Step 4: Run, expect PASS**: `go test ./internal/phone/...`
- [ ] **Step 5: Commit** `feat(phone): carry the call id on the leg to Stream`

### Task 4: Store and lines: one trunk per customer and vendor

**Files:**
- Create: `acceleration/migrations/20261106120000_shared_trunks.sql`
- Create: `acceleration/internal/store/streamtrunks.go`, test `streamtrunks_test.go` (`StoreSuite`; add `stream_trunks` to the `TRUNCATE` list in `store_test.go:119`)
- Create: `acceleration/internal/phone/lines.go`, test in `apps_integration_test.go` (`AppsSuite`)
- Modify: `acceleration/internal/store/models.go` (`PhoneNumber`, `CallResource`), `callresources.go`, `store.go` (`AttachNumber`)

**Migration:**
- `stream_trunks`: `id TEXT PRIMARY KEY` (Stream's trunk id), `customer_id TEXT NOT NULL`, `stream_app_pk BIGINT` (null = the deployment's app), `vendor TEXT NOT NULL`, `numbers TEXT[] NOT NULL`, `bridge_uri TEXT NOT NULL`, `bridge_username TEXT NOT NULL`, `bridge_password_sealed BYTEA NOT NULL`, `bridge_password_kek_version INT NOT NULL`, `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`; index on `(customer_id, stream_app_pk, vendor)`.
- `phone_numbers.attached_at TIMESTAMPTZ`; backfill `attached_at = purchased_at WHERE stream_trunk_id IS NOT NULL AND released_at IS NULL`.
- `call_resources`: drop the `trunk_id` primary key, add `id BIGSERIAL PRIMARY KEY` and `shared BOOLEAN NOT NULL DEFAULT false`. It runs on shards too (`shards.go:193`).

**Interfaces:**
- Store produces:
  - `type TrunkGroup struct { CustomerID string; StreamAppPK int64; Vendor string }`
  - `type StreamTrunk struct` for the table, `bun:"table:stream_trunks"`, fields named as the columns.
  - `func (s *Store) InTrunkGroup(ctx context.Context, group TrunkGroup, fn func(ctx context.Context, tx *TrunkGroupTx) error) error`: one `RunInTx` holding `pg_advisory_xact_lock(hashtextextended(?, 0))` on `"stream_trunks:" + customer + ":" + app + ":" + vendor`, the pattern in `connectors.go:231`.
  - `TrunkGroupTx` methods: `Trunks() ([]StreamTrunk, error)` (the group's), `SaveTrunk(StreamTrunk) error` (upsert), `DeleteTrunk(id string) error`, `SetLine(e164, trunkID, routeID string, bridge NumberBridge) error` (writes `stream_trunk_id`, `stream_route_id`, `stream_app_pk`, the bridge columns on the held number), `ClearLine(e164 string) error`.
  - `type NumberBridge struct { URI, Username string; PasswordSealed []byte; KEKVersion int }`.
  - `PhoneNumber.AttachedAt *time.Time`; `AttachNumber` sets it; `ReleaseNumber` clears it.
  - `CallResource.Shared bool`; `RecordCallResource` unchanged in signature; `func (s *Store) DeleteCallResource(ctx context.Context, callType, callID, trunkID string) error`.
- Phone produces (`lines.go`):
  - `type Line struct { TrunkID, RouteID string; Bridge Bridge }` (`Bridge` already `WithNumber`, password opened).
  - `const lineCallTemplate = "{{#if x_stream_call_id}}{{x_stream_call_id}}{{else}}{{uuid}}{{/if}}"`; `inboundCallTemplate` goes.
  - `const trunkCapacity = 50`.
  - `func (s *Service) line(ctx context.Context, stream *Stream, pin int64, held store.PhoneNumber) (Line, error)`:
    1. Held number already on a trunk in this app with a `stream_trunks` row: return its line, nothing at Stream.
    2. Held number on a trunk of its own with no row (attached before this change): adopt it. Save a `stream_trunks` row from the number's bridge columns, `UpdateRoute` its rule to `lineCallTemplate`, keep its ids.
    3. Otherwise, inside `InTrunkGroup`: take the trunk with the fewest numbers under 50 and `UpdateTrunk` it with the number added, or `CreateTrunk` (`Name: "phone-" + vendor + "-" + customer`, allowlist from `allowlistFor`) and seal its password. Then `CreateRoute{Name: "phone-" + e164, CallID: lineCallTemplate, Custom: {NumberKey: e164}}`. Then `SaveTrunk` and `SetLine`.
    - A failure after Stream changed anything undoes it inside the same call: the rule is deleted, the number is taken back off the trunk's list (or the new trunk deleted), and nothing is stored.
    - Needs `s.sealer`; returns `ErrNoSealer` without it (the caller decides the fallback, Task 5).

- [ ] **Step 1: Write the failing store tests** in `streamtrunks_test.go`: `TestATrunkGroupSeesOnlyItsOwnTrunks` (other customer, other vendor, other app invisible), `TestTheDeploymentsAppIsAGroupOfItsOwn` (app 0), `TestTwoTransactionsOnOneGroupRunOneAfterTheOther` (two goroutines; the second sees the first's trunk), `TestSettingALineIsReadBackOnTheNumber`, `TestReleasingANumberClearsItsAttachedAt`, `TestTwoCallResourcesMayShareATrunk`.
- [ ] **Step 2: Run, expect FAIL**: `ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router_test?sslmode=disable' go test -tags integration -run TestStoreSuite ./internal/store`
- [ ] **Step 3: Implement** the migration and the store.
- [ ] **Step 4: Run, expect PASS** (same command), and `go test -tags integration -run TestShardedStoreSuite ./internal/store`.
- [ ] **Step 5: Write the failing line tests** in `apps_integration_test.go` (call `s.service.line` directly; it is in-package): `TestTwoNumbersOfOneVendorShareATrunk` (`s.own.Trunks()` has one id, `TrunkNumbers` both), `TestTheFiftyFirstNumberOpensASecondTrunk`, `TestCustomersNeverShareATrunk`, `TestVendorsNeverShareATrunk`, `TestEachNumberHasARuleOfItsOwnThatReadsTheHeader` (`Rule(id)` call id `== lineCallTemplate`, called numbers `[e164]`), `TestANumberAttachedBeforeSharedTrunksIsAdoptedAsItIs` (seed a number with `StreamTrunkID` and bridge columns, no row: same trunk id, rule updated, no new trunk), `TestTwoNumbersJoiningAtOnceMakeOneTrunk`, `TestARefusedRuleLeavesTheNumberOffTheTrunk` (fake refuses the rule: `TrunkNumbers` without the number, `store.Number` has no `StreamTrunkID`).
- [ ] **Step 6: Run, expect FAIL**: `go test -tags integration -run TestAppsSuite ./internal/phone`
- [ ] **Step 7: Implement `lines.go`.**
- [ ] **Step 8: Run, expect PASS** (same command).
- [ ] **Step 9: Commit** `feat(phone): shared trunks, a line per number`

### Task 5: Calls and transfers on lines

**Files:**
- Modify: `acceleration/internal/phone/service.go` (`Call` 618-777, `Transfer` 1063-1178, `ReleaseCall` 945-966)
- Test: `apps_integration_test.go`, `integration_test.go` (`TransferIntegrationSuite`)

**Interfaces:**
- Consumes: `line`, `StreamLegOf`, `Outbound.CallID`, `Stream.CreateCall`, `CallResource.Shared`, `DeleteCallResource`.
- Behaviour: when `StreamLegOf(provider) == StreamLegShared` and `s.sealer != nil`:
  - `Call`: `line()` for `request.From`; `CreateCall(ctx, callType, callID, request.Custom)` when `request.Custom` is not empty; `outbound.Bridge = line.Bridge`, `outbound.CallID = callID`; `RecordCallResource{TrunkID: line.TrunkID, RouteID: line.RouteID, Shared: true, …}` before `provider.Dial`, deleted with `DeleteCallResource` when the dial fails.
  - `Transfer`: the same, with `CallID: request.CallID` and no `CreateCall`.
  - Otherwise both run today's per-call code unchanged.
  - `ReleaseCall`: shared rows count toward `gate.Talked` like any row, and are never unwired.

- [ ] **Step 1: Write the failing tests:**
  - `TestTwoCallsFromOneNumberAtOncePlaceBoth` with `s.own.OneTrunkPerNumber()`: both `Call`s succeed, `s.own.Trunks()` has one trunk (AI-1279).
  - `TestACallFromAnAttachedNumberPlaces` with `OneTrunkPerNumber()` (AI-894).
  - `TestSixtyNumbersPlacingCallsStayUnderTheTrunkCap` with `s.own.CapTrunks(25)`: 60 numbers of one vendor each place a call; all succeed, 2 trunks (AI-1318).
  - `TestACallsCustomIsOnItsStreamCall`: `fake.Call("agent", placed.CallID)["custom"]` holds `request.Custom`.
  - `TestTheVendorIsGivenTheSessionAsTheCallID`: the stub vendor's last `Outbound.CallID == placed.SessionID` (the `stub` vendor implements `StreamLegger` shared for this suite; a second stub stays per-call).
  - `TestACallOnAVendorWithoutTheHeaderGetsATrunkOfItsOwn` (per-call stub: a `call-<id>` trunk, as today).
  - `TestWithoutAKeyACallStillPlacesOnATrunkOfItsOwn` (`sealer` nil).
  - `TestAFailedDialLeavesNoCallResource`.
  - `TestEndingASharedCallKeepsItsTrunkAndCountsItsMinutes`: `ReleaseCall`; trunk and rule still there, the gate saw the talk time.
  - Transfer: `TestATransferFromANumberWithALiveCallPlaces`.
- [ ] **Step 2: Run, expect FAIL**: `go test -tags integration -run 'TestAppsSuite|TestTransferIntegrationSuite' ./internal/phone`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the whole package: `go test -tags integration ./internal/phone/...`
- [ ] **Step 5: Commit** `fix(phone): calls share their number's trunk`

### Task 6: Attach and release on lines

**Files:**
- Modify: `acceleration/internal/phone/service.go` (`Attach` 429-544, `Release` 373-396)
- Modify: `acceleration/internal/api/phone.go:716` (`stream_trunk_id` only when `AttachedAt != nil`)
- Test: `apps_integration_test.go`, `acceleration/internal/api/phone_test.go`

**Interfaces:**
- Consumes: `line`, `TrunkGroupTx.ClearLine`, `Stream.UpdateTrunk`.
- `Attach`: `line()`, then `ConfigureInbound` with `line.Bridge`, then `AttachNumber` (sets `AttachedAt`). Repeat attach is a no-op when `AttachedAt != nil` in the same app (replaces the `StreamTrunkID` check at 461). Without a sealer it keeps today's trunk of its own.
- `Release`: inside `InTrunkGroup`: `Stream.DeleteRoute`, then `Stream.UpdateTrunk` without the number, or `Stream.DeleteTrunk` and `tx.DeleteTrunk` when it was the last; `tx.ClearLine`. A number on a trunk of its own (no row) is unwired as today.
- Moving apps (registration, `TestReattachingAfterRegistrationMovesTheTrunk`): the old app's line is released the same way, then a line is made in the new app.

- [ ] **Step 1: Write the failing tests:**
  - `TestAttachingANumberThatAlreadyPlacedCallsKeepsItsTrunk` (one trunk, one rule).
  - `TestReleasingOneOfTwoNumbersKeepsTheTrunkForTheOther` (`TrunkNumbers` = the other, its rule still there).
  - `TestReleasingTheLastNumberDeletesTheTrunk`.
  - `TestAFreedPlaceOnAFullTrunkIsTakenBeforeASiblingIsOpened` (50 numbers, release one, add one: still one trunk).
  - Keep passing: `TestAttachingAnAttachedNumberAgainChangesNothing`, `TestReattachingAfterRegistrationMovesTheTrunk`, `TestReleasingANumberDeletesItsTrunkAndRouteInItsApp` (rename to `…DeletesItsRouteAndItsPlaceOnTheTrunk` if its assertion changes).
  - api `PhoneSuite`: `TestANumberThatOnlyPlacedCallsIsNotListedAsAttached` (no `stream_trunk_id` in `GET /v1/phone/numbers`).
- [ ] **Step 2: Run, expect FAIL**: `go test -tags integration -run TestAppsSuite ./internal/phone` and `go test -tags integration -run TestPhoneSuite ./internal/api`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (same commands).
- [ ] **Step 5: Commit** `feat(phone): attach and release numbers on shared trunks`

### Task 7: The hook skips placed calls; old per-call trunks are swept

**Files:**
- Modify: `acceleration/internal/api/callhooks.go` (`dispatchArrivingCall` 112-129)
- Modify: `acceleration/internal/phone/service.go` (new `SweepCallResources`), `acceleration/internal/store/callresources.go`
- Create: `acceleration/cmd/router/phonesweep.go`; wire it in `cmd/router/main.go` beside `startEpisodeSweeper` (1201)
- Test: `acceleration/internal/api/callhooks_test.go` (`CallHooksSuite`), `apps_integration_test.go`

**Interfaces:**
- `dispatchArrivingCall`: before the `NumberKey` lookup, a call with any row in `CallResourcesInApp(ctx, origin.scope(), callType, callID)` is the router's own and is not dispatched (logged at debug).
- `func (s *Store) StaleCallResources(ctx context.Context, before time.Time) ([]CallResource, error)`: rows with `shared = false` and `created_at < before`, deleted as they are returned (like `ReleaseCallResourcesInApp`).
- `func (s *Service) SweepCallResources(ctx context.Context, before time.Time) (int, error)`: unwires each, returns how many.
- `startPhoneSweeper(ctx context.Context, service *phone.Service)`: every 10 minutes, `before = now - 6h` (a call longer than six hours is not a campaign call; the session's own limits end it far sooner). Started only when the phone service has a store.

- [ ] **Step 1: Write the failing tests:**
  - `TestAnArrivingCallTheRouterPlacedIsNotDispatched` (seed a `call_resources` row, post `call.session_started` with `phone_number` in custom: no worker assigned).
  - `TestAnArrivingCallToANumberIsStillDispatched` (no row: assigned, as today).
  - `TestAPerCallTrunkWhoseCallNeverEndedIsSweptAfterSixHours` (row backdated 7 h: trunk and rule gone at the fake).
  - `TestASharedCallsResourceIsNeverSwept`.
- [ ] **Step 2: Run, expect FAIL**: `go test -tags integration -run TestCallHooksSuite ./internal/api` and `-run TestAppsSuite ./internal/phone`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then `go build ./... && go vet ./...`
- [ ] **Step 5: Commit** `fix(phone): sweep per-call trunks the end hook missed`

### Task 8: Live verification, docs, changelog

**Files:**
- Modify: `.claude/skills/phonecheck/SKILL.md` (step 1.5: an attached number may place outbound calls now)
- Modify: `acceleration/internal/phone/README.md` if it describes per-call trunks (check)
- Modify: `CHANGELOG.md` (check the top heading; `# Unreleased` above a tagged one)

- [ ] **Step 1: Live, with the phonecheck skill** on the branch's router: two outbound calls at once from one Twilio number, both `outcome=ok`; one outbound from an attached Telnyx number, `outcome=ok`; two inbound calls to that number at once, each answered by a worker in a call of its own (AI-988; the fake cannot render templates, so this is the only proof). Record the session ids.
- [ ] **Step 2: Stream side:** the test app's trunks = one per vendor used (`go run ./cmd/phone hooks` is not it; list trunks with the SDK's `ListSIPTrunks` in a one-off `go run`).
- [ ] **Step 3: Docs and changelog.** Changelog under **Bug Fixes**: parallel calls from one number (AI-1279), calls from attached numbers (AI-894), trunks no longer leak toward Stream's cap (AI-1318).
- [ ] **Step 4: Full check:** `cd acceleration && go build ./... && go vet ./... && go test ./... && go test -tags integration ./internal/phone/... ./internal/store ./internal/api`
- [ ] **Step 5: Commit** `docs(phone): shared trunks`
