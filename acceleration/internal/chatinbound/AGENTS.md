# internal/chatinbound

Stream Chat's own delivery of messages to the router. `chatinbound.go` checks a delivery's signature. `worker.go` answers the deliveries queued in `chat_inbound`, on every router.

## Flow

```
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

- **One row is one answer.** No try of a row answers it a second time, on any router. Example: a peer answers and its reply is lost; the next try finds the row `done` and answers nothing (`TestAQueuedDeliveryAPeerAnsweredIsNotAnsweredAgainWhenItsReplyWasLost`).
- **The session and dispatch paths settle the row as the answer starts.** `answerOnce` moves the row from `queued` to `done` (`store.StartChatInboundAnswer`) before the answer. A router that stops mid-answer leaves the message answered at most once. This is the trade-off: at most once, not at least once.
- **The thread path is claimed by its turn.** `answerThread` claims the message's turn (`store.ClaimChannelThreadMessage`, `store.ClaimTurn`). The row stays `queued` for the worker to finish. A second try finds the turn taken and answers nothing.
- **A row goes back to the queue only on a failure that proves nothing took it.** `unanswered` moves it from `done` to `queued` (`store.ReleaseChatInboundAnswer`). Example: no dispatch worker is connected. A failure that leaves it open keeps the row settled. Example: the node with the worker did not reply in time (`node.ErrNoReply`); it may have placed the message (`TestAQueuedDeliveryAnotherNodePlacedIsNotPlacedAgainWhenItsReplyWasLate`).
- **A hook settles nothing.** A webhook delivery is never tried again, so it has no row.
- **Close waits for the answers in flight.** It records what came of each, so no row waits for its lease to run out.

## Tests

From `acceleration/`:

```bash
go test -race ./internal/chatinbound
go test -tags integration -race ./internal/chatinbound
go test -tags integration -run 'TestChatInbound' ./internal/api
go test -tags integration -run TestStoreSuite -testify.m Chat ./internal/store
```
