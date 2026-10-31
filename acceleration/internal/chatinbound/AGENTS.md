# internal/chatinbound

Stream Chat's own delivery of messages to the router. `chatinbound.go` checks a delivery's signature. The endpoint (`api.receiveChatInbound`) queues it in `chat_inbound`. `worker.go` answers the queued deliveries, on every router.

## Flow

```
endpoint                                    api.receiveChatInbound
  signature, app served here, message.new, addressed, fresh (app mode)
  one tx: QueueFirstChatInbound
    hook_deliveries <app>:message:<id>      the webhook's key (deliveryKey)
    chat_inbound row
  committed: 200, Kick wakes this router's worker
  key recorded before: 200, nothing queued  the webhook or an earlier delivery answers it
  store error or no store: 503              Stream Chat delivers it again

worker (every router)                       Worker.answerDue
  ClaimChatInbound: due rows, lease 5 min, SKIP LOCKED
  Answer = api.AnswerChatInbound -> routeArrivingMessage
    thread channel     answerThread claims the turn; row stays queued
    session here       answerOnce: queued -> done, then answer
    worker here        answerOnce: queued -> done, then AssignMessage
    session or worker  hand over; the peer runs the same steps
      on a peer
  nil: FinishChatInbound   error: again after 2s, 10s, 30s, 2min, 10min
  out of waits: failed
done rows deleted after 24h; failed rows kept 30 days
```

## Rules

- **200 means committed.** The endpoint answers 200 only after the row and its key commit, and 503 when they could not be written. Example: the database is down; Stream Chat delivers again and the retry is answered (`TestADeliveryThatCouldNotBeQueuedIsMadeAgainAndAnswered`).
- **One key per message, whoever brings it.** The endpoint and the message webhook record the same key, `<app>:message:<id>`, so the second of the two acts on nothing. In deployment mode the webhook records it only while Stream Chat deliveries are on (`TestAMessageTheWebhookBroughtIsNotQueuedWhenDelivered`, `TestTheWebhookOfAMessageDeliveredFirstIsNotActedOn`).
- **The worker never reads the key.** The key keeps a second delivery out of the queue; the row is what the worker answers. Example: the router stops after the 200; the next worker answers the row (`TestADeliveryTakenBeforeTheRouterStoppedIsAnsweredOnceAWorkerStarts`).
- **One row is one answer.** No try of a row answers it a second time, on any router. Example: a peer answers and its reply is lost; the next try finds the row `done` and answers nothing (`TestAQueuedDeliveryAPeerAnsweredIsNotAnsweredAgainWhenItsReplyWasLost`).
- **The session and dispatch paths settle the row as the answer starts.** `answerOnce` moves the row from `queued` to `done` (`store.StartChatInboundAnswer`) before the answer. A router that stops mid-answer leaves the message answered at most once. This is the trade-off: at most once, not at least once.
- **The thread path is claimed by its turn.** `answerThread` claims the message's turn (`store.ClaimChannelThreadMessage`, `store.ClaimTurn`). The row stays `queued` for the worker to finish. A second try finds the turn taken and answers nothing.
- **A row goes back to the queue only on a failure that proves nothing took it.** `unanswered` moves it from `done` to `queued` (`store.ReleaseChatInboundAnswer`). Example: no dispatch worker is connected. A failure that leaves it open keeps the row settled. Example: the node with the worker did not reply in time (`node.ErrNoReply`); it may have placed the message (`TestAQueuedDeliveryAnotherNodePlacedIsNotPlacedAgainWhenItsReplyWasLate`).
- **A hook settles nothing.** A webhook delivery is never tried again, so it has no row.
- **The stop signal stops the claims.** `Start` takes the router's stop signal (cmd/router: the signal context, as `scheduling.Scheduler.Start`), so a draining router claims no more rows; the rows still queued wait for another router. The answers already running go on (`TestAfterTheStopSignalAWorkerFinishesItsAnswerAndClaimsNoOther`), and cmd/router's drain waits for them (`Running`, counted in `running` as the scheduled deliveries are) until the drain runs out (`TestRunningCountsTheAnswersInFlight`, `TestDrainSuite`).
- **Close waits for the answers in flight.** It records what came of each, so no row waits for its lease to run out.

## Tests

From `acceleration/`:

```bash
go test -race ./internal/chatinbound
go test -tags integration -race ./internal/chatinbound
go test -tags integration -run 'TestChatInbound' ./internal/api
go test -tags integration -run TestStoreSuite -testify.m Chat ./internal/store
```
