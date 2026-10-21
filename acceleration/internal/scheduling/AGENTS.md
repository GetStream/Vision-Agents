# internal/scheduling

Scheduled tasks: something the agent agreed to do for one end user at a wall-clock time in
that user's zone, once or on a repeat. This package holds the two pure parts and the worker:
schedules (`scheduling.go`, `resolve.go`, `describe.go`, `quiet.go`: parse, validate, expand,
describe, quiet hours) and the `Scheduler` (`scheduler.go`), which claims due occurrences from
the store and hands each to a `DeliverFunc`. It imports `store` and nothing that delivers; the
delivery is `api.Server.DeliverScheduledTask`, handed in at `cmd/router/main.go:1265`. What the
feature is for and what is not done: `.factory/features/scheduled-tasks.md`. How to use and
debug it: `.claude/skills/scheduled-tasks/SKILL.md`. Paths are under `acceleration/`.

## Code map

```
 model calls a tool                       backend calls REST
 internal/session/schedule_tools.go       internal/api/scheduled_tasks.go
 (schedule_task, list_, update_, cancel_) (/v1/agents/scheduled-tasks ...)
              \                                 /
               \   scheduling.Parse, ResolveZone, CheckMove, SleptThrough
                \                             /
                 v                           v
        internal/store/scheduled_tasks.go  (tables: scheduled_tasks, scheduled_task_runs)
        migrations/20261105120000_scheduled_tasks.sql
        migrations/20261105130000_scheduled_tasks_wake.sql  (scheduled_tasks_wake_idx)
                 |
                 |  ClaimDueScheduledTasks (FOR UPDATE SKIP LOCKED, lease, claim_id)
                 v
        internal/scheduling/scheduler.go   Scheduler.run: missed? -> deliver with retries
                 |                         -> FinishScheduledTaskRun | PostponeScheduledTask
                 |  DeliverFunc
                 v
        internal/api/scheduled_delivery.go DeliverScheduledTask
                 |   quiet hours? -> Postponed
                 |   session live here | on a peer (POST /internal/scheduled-tasks/deliver)
                 |   | ended: reopen from its row | none yet: open one
                 v
        deliver_to, in order, first that lands:
          conversation  Session.FollowUp(command id "st_<run id>")
          sms | whatsapp | imessage   same turn's words -> channels.Service.Notify
          phone         phone.Service.Call -> call session -> task as first turn
```

## Terms

One word, one meaning, in code, docs and API alike.

| Term | What it is | In code | Example |
| --- | --- | --- | --- |
| task | One agreement with one user: title, instructions, schedule, zone, `deliver_to`, `quiet_hours`, state `active`, `paused`, `completed` or `failed` | `store.ScheduledTask`, table `scheduled_tasks`, API `ScheduledTask` | "Stretch", every weekday at 09:00, Europe/Amsterdam |
| occurrence | One instant the schedule names | `Schedule.Next`, `Schedule.Upcoming`; the task's `next_run_at` is the next one | Mon 26 Oct 2026 09:00 CET |
| run | The record of one occurrence that came due, or of one run by hand: `running`, `delivered`, `failed` or `missed`, with `error` a reason code. Its `prompt` is the instructions as they were at the claim | `store.ScheduledTaskRun`, table `scheduled_task_runs`, unique per (`customer_id`, `task_id`, `scheduled_for`) for `trigger = schedule` | run `scheduled_for` 2026-10-26T09:00:00+01:00, `state` `delivered`, `delivered_to` `conversation` |
| claim, lease | A worker's hold on a task: `claim_id` fences every later write of that worker, `lease_until` says when another worker may take it. Lease 1 min, renewed every 10 s | `ClaimDueScheduledTasks`, `RenewScheduledTaskLease`, `scheduledLease`, `renewLeaseEvery` (`scheduler.go:19`, `:22`) | A pause clears `claim_id`; the next renewal finds no row and stops the delivery |
| held | An occurrence that came due in quiet hours: the claim stays, the lease moves to the hold's end, the run stays `running` with `postponed_until` | `Postponed`, `PostponeScheduledTask`, `ScheduledTaskRun.PostponedUntil` | Due 23:30 with quiet hours 22:00–07:00: held until 07:00, delivered then |
| missed | A run not delivered because it was too late or the task slept through it. Not a failure: the failure streak is left as it is | `ScheduledTaskRunMissed`, `Missed`, `Scheduler.missed`, `SleptThrough` | A daily task the router could not reach for 3 h: the two oldest occurrences are `missed`, the latest is delivered |
| reason | The short code a failed or missed run records; the full error goes to the logs only | `Reason*` constants (`scheduler.go:105`), `Failure.Reason` | `user_on_call`, `not_answered`, `task_paused` |
| final | A failure another attempt would only repeat, or that would ring the user again: not retried | `Failure.Final`, `undeliveredFinally` | `not_answered`, `already_called` |
| quiet hours | The part of the user's local day an occurrence is held through: `{start, end}` as "22:00", read in the task's zone; an end before the start wraps midnight | `QuietEnd` (`quiet.go:46`), `ScheduledTaskQuietHours` | `{"start": "22:00", "end": "07:00"}` |
| schedule, RRULE subset | `at` or `in`, an optional `repeat` (FREQ DAILY, WEEKLY, MONTHLY, YEARLY; INTERVAL; BYDAY; BYMONTHDAY; COUNT ≤ 1000) and `until`, an inclusive local date. `start_local` is stored as the wall clock with no zone | `Input`, `Schedule`, `Parse`, `FromStored`, `parseRepeat` (`scheduling.go:436`) | `at: "09:00"`, `repeat: "FREQ=MONTHLY;BYDAY=-1FR"`: last Friday of every month |
| zone resolution | Which IANA zone a new task takes, first match wins: the call's argument, the session's, `users.timezone`, the config's `default_timezone`; none is `timezone_unknown`. `Local` is refused | `ResolveZone` (`scheduling.go:153`), `LoadZone` | Session opened with `timezone: "America/New_York"`, no argument: the task is in New York time |
| due | When an occurrence came due, or when the hold on it ended; lateness counts from it | `scheduledDelivery.Due`, `Scheduler.missed` | Held until 07:00: a 07:05 delivery is 5 min late, not 7 h 35 min |
| command id | The id the follow-up turn is sent under, so a conversation runs it once | `"st_" + run.ID` (`scheduled_delivery.go:161`); a call's turn adds `_call` | `st_01JB...` |

## Rules

Each rule names the test that fails when it breaks. Suites: `SchedulerSuite`
(`internal/scheduling/scheduler_test.go`), `StoreSuite` (`internal/store/scheduled_tasks_test.go`),
`ScheduledTasksSuite` (`internal/api/scheduled_tasks_test.go`, `scheduled_phone_test.go`).
All three need the integration tag (see Run the tests).

- **One occurrence, one claim, one run.** Claims run in one transaction under
  `FOR UPDATE SKIP LOCKED`, against the database's clock, and insert the run in it.
  `TestSchedulerSuite/TestTwoWorkersDeliverAnOccurrenceOnce`,
  `TestStoreSuite/TestClaimingGivesEachDueTaskToOneWorker`,
  `TestStoreSuite/TestAClaimInsertsOneRunPerOccurrence`.
- **Every write after a claim goes through its `claim_id`.** A finish, renewal or hold under a
  claim a pause, edit or delete took changes nothing on the task.
  `TestStoreSuite/TestAFinishUnderAStolenClaimDoesNotAdvanceTheTask`,
  `TestStoreSuite/TestARenewalExtendsOnlyTheClaimItHolds`,
  `TestStoreSuite/TestAPostponeUnderAStolenClaimHoldsNothing`.
- **A lost claim stops the delivery.** `TestSchedulerSuite/TestPausingATaskStopsItsDeliveryInFlight`,
  `TestSchedulerSuite/TestAnEditedTaskIsNotRetriedByTheWorkerThatLostIt`.
- **A run taken over keeps its id, so its command id.** A stale run is reclaimed under the same
  row; after three workers died on it, it fails with `delivery_failed`
  (`maxScheduledTaskRunAttempts`). `TestStoreSuite/TestAStaleRunIsReclaimedUnderTheSameRun`,
  `TestStoreSuite/TestAnOccurrenceWhoseWorkersKeepDyingIsFailedAtTheAttemptCap`.
- **An occurrence holds a slot 20 min at most** (`occurrenceBudget`), and running out is
  `timed_out`. `TestSchedulerSuite/TestAnOccurrenceThatOutlastsItsBudgetFails`,
  `TestSchedulerSuite/TestAnOccurrenceThatOutlastsItsBudgetIsTimedOutEvenWhenDeliveryAnswersWithACodedFailure`.
- **Late is missed, never replayed.** A one-shot delivers up to 24 h late, a repeat up to 1 h
  (`oneShotGrace`, `repeatGrace`); older ones get a `missed` run each.
  `TestSchedulerSuite/TestAOneShotMoreThanADayLateIsMissedNotDelivered`,
  `TestSchedulerSuite/TestARecurringTaskSkipsOccurrencesOlderThanAnHour`,
  `TestSchedulerSuite/TestReplayingAMissedRunKeepsTheFailureStreak`.
- **Five failed occurrences in a row fail the task** (`maxConsecutiveFailures`).
  `TestSchedulerSuite/TestAFifthFailureInARowFailsTheTask`.
- **A final failure is not retried.** `TestSchedulerSuite/TestAFinalFailureIsNotRetried`,
  `TestScheduledTasksSuite/TestAnOccurrenceIsFinalOnlyWhenEveryChannelsFailureIs`.
- **Quiet hours hold, never fail, and age counts from the hold's end.** A run made by hand is
  not held. `TestSchedulerSuite/TestAHeldTaskIsDeliveredWhenTheHoldEnds`,
  `TestSchedulerSuite/TestAHeldOccurrenceIsCountedFromTheHoldAndNotFromWhenItCameDue`,
  `TestScheduledTasksSuite/TestAQuietTaskIsHeldUntilItsQuietHoursEnd`.
- **A pause or edit writes a held run `missed` (S4).** An edited active one-shot comes due
  again as it now reads; one being delivered keeps its claim and run.
  `TestStoreSuite/TestPausingATaskWritesItsHeldOccurrenceMissed`,
  `TestScheduledTasksSuite/TestPausingATaskWritesTheOccurrenceItHeldMissed`,
  `TestStoreSuite/TestAnEditedOneShotWhoseOccurrenceWasHeldComesDueAgainAsItNowReads`,
  `TestStoreSuite/TestAnEditedOneShotBeingDeliveredKeepsItsClaimAndItsRun`.
- **A paused one-shot resumed after its time is completed with one `missed` run, error
  `task_paused` (AI-1602)**, over the API and the tool alike; the tool result names the time in
  the task's zone. `TestAPausedOneShotSleepsThroughItsTimeAsAMissedRun` (unit),
  `TestScheduledTasksSuite/TestAOneShotResumedAfterItsTimeRecordsTheOccurrenceAsMissed`,
  `TestScheduledTasksSuite/TestAOneShotTheAgentResumesAfterItsTimeRecordsTheOccurrenceAsMissed`,
  `TestScheduledTasksSuite/TestAOneShotTheAgentResumesLateNamesTheMissedTimeInTheTasksZone`,
  `TestScheduledTasksSuite/TestResumingALateOneShotTwiceRecordsItsOccurrenceOnce`.
- **A call rings once, and never more than 15 min late (S5).** Too late is `missed` without a
  call; nobody answering is `not_answered`, final.
  `TestScheduledTasksSuite/TestAPhoneTaskMoreThanAQuarterHourLateIsMissedWithoutACall`,
  `TestScheduledTasksSuite/TestACallNobodyPicksUpIsNotAnswered`,
  `TestScheduledTasksSuite/TestAnOccurrenceThatRangAlreadyIsNotRungAgainNorRetried`.
- **A repeat `Describe` would misstate is refused (S18)**; a stored one still runs.
  `TestARepeatItsDescriptionWouldMisstateIsRefused` (unit and `ScheduledTasksSuite`),
  `TestAStoredRepeatNowRefusedStillRuns` (unit).
- **The active-task limit counts per user per config (S20).**
  `TestStoreSuite/TestTheActiveLimitCountsTheTasksOnTheSameConfigOnly`,
  `TestScheduledTasksSuite/TestTheActiveLimitCountsTheTasksOnTheirOwnAgentOnly`.
- **A wall-clock time stays put across DST.** `TestADailyTaskHoldsItsWallClockAcrossDaylightSaving`,
  `TestTheSpringGapMovesForwardInAmsterdam`, `TestTheFallOverlapTakesTheFirstOccurrenceInAmsterdam`
  (unit).
- **No zone is a refusal, not a guess.** `TestResolveZoneWithoutAnyZoneAsks` (unit),
  `TestScheduledTasksSuite/TestATaskIsRefusedInWordsWhenNoZoneIsKnown`.
- **A run moved earlier lands 15 min from now at the soonest** (`MinMove`).
  `TestAMoveLandsNoSoonerThanMinMoveUnlessItKeepsTheCurrentRun` (unit),
  `TestScheduledTasksSuite/TestANextSoonerThanAQuarterHourIsRefused`.
- **The tools go only to verified users with scheduling on, outside incognito.**
  `TestScheduledTasksSuite/TestGuestsAndAnonymousCallersAreNotOfferedTheTools`,
  `TestScheduledTasksSuite/TestAnIncognitoSessionIsNotOfferedTheTools`,
  `TestScheduledTasksSuite/TestAnAgentWithoutSchedulingOnIsNotOfferedTheTools`,
  `TestScheduledTasksSuite/TestACallersOwnScheduleTaskKeepsTheName`.
- **Another user's task is a 404.** `TestScheduledTasksSuite/TestTheOwnerReachesATaskItsBackendMadeAndAnotherUserDoesNot`.
- **A scheduled turn is never on the user's day.** `TestScheduledTasksSuite/TestADueTaskIsNotCountedOnItsUsersDay`.
- **A call in progress is waited out**, and a session on another node is handed over.
  `TestScheduledTasksSuite/TestATaskThatComesDueOnACallWaitsForTheCallToEnd`,
  `TestScheduledTasksSuite/TestADueTaskIsHandedToTheNodeHoldingItsConversation`.
- **The reply is written once; channels carry its words.**
  `TestScheduledTasksSuite/TestARetriedOccurrenceTextsTheReplyItAlreadyHas`,
  `TestScheduledTasksSuite/TestAWhatsAppTaskOutsideItsWindowLandsInTheConversation`,
  `TestScheduledTasksSuite/TestATaskForSomebodyWhoOptedOutRunsNoTurnForThem`.
- **Instructions are data, not orders.** `TestInstructionsCannotCloseTheirScheduledTaskBlock`
  (`internal/api`, unit).
- **The worker sleeps on the wake index.** `TestStoreSuite/TestTheNextWakeIsReadOffItsIndex`,
  `TestScheduledTasksMigrationSuite/TestTheWakeIndexAppliesComesOffAndAppliesAgain`.
- **A draining router finishes what it claimed.** `TestDrainSuite/TestTheDrainWaitsForAScheduledDeliveryInFlight`
  (`cmd/router`), `TestSchedulerSuite/TestAStoppedWorkerClaimsNothingMoreButFinishesItsDelivery`.
- **Config delete pauses, session delete deletes, data move carries.**
  `TestScheduledTasksSuite/TestDeletingAConfigPausesItsTasks`,
  `TestScheduledTasksSuite/TestDeletingASessionDeletesItsTasks`,
  `TestDataMoveSuite/TestScheduledTasksMoveWithTheirRuns`.

## Run the tests

The unit tests need nothing:

```bash
cd acceleration
go test ./internal/scheduling/
go test -run 'TestAPeerIsGivenLongEnough|TestInstructionsCannotClose|TestPlainInstructions|TestATurnThatSaidNothing|TestATaskIsRenderedWithout' ./internal/api/
```

The suites need Postgres and Redis (`docker compose up -d postgres redis` at the repo root:
`:55432`, `:56379`). `internal/testenv` loads `internal/config/testing.yaml` over the shell,
so `testing.yaml` wins over `ROUTER_POSTGRES_DSN` (`TestTheTestingFileWinsOverTheEnvironment`):

1. Point `testing.yaml` at a database of your own whose name ends in `_test`, such as
   `model_router_test_<slug>_test`. Never commit that edit. `StoreSuite` drops the schema of
   the database it gets, and `SchedulerSuite` migrates and writes `ROUTER_POSTGRES_DSN` as is.
2. Keep `redis.addr: localhost:56379` there: `ScheduledTasksSuite` skips without both
   `ROUTER_POSTGRES_DSN` and `ROUTER_REDIS_ADDR` (`internal/api/router_suite_test.go:367`).
3. Run one package at a time (`-p 1`), as the `parallel-agents` skill says:

```bash
cd acceleration
grep dsn internal/config/testing.yaml        # your own *_test database
go test -tags integration -p 1 -count=1 -run TestSchedulerSuite ./internal/scheduling/
go test -tags integration -p 1 -count=1 -run 'TestStoreSuite|TestScheduledTasksMigrationSuite' ./internal/store/
go test -tags integration -p 1 -count=1 -run TestScheduledTasksSuite ./internal/api/
go test -tags integration -p 1 -count=1 -run TestDrainSuite ./cmd/router/
```

Narrow to one test with `-run 'TestScheduledTasksSuite/TestACallNobodyPicksUpIsNotAnswered'`.
Known flakes: AI-1555, AI-1566.
