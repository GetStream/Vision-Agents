# Campaigns

## Asked for

An API for outbound calling: define the concurrency, point at an agent config, and give each
person you are calling their own instructions.

## What exists

[internal/campaign](../../acceleration/internal/campaign) is the dialer. Three tables hold it:
`campaigns`, `campaign_contacts` and `campaign_attempts`
([store/campaigns.go](../../acceleration/internal/store/campaigns.go),
[store/campaign_dispatch.go](../../acceleration/internal/store/campaign_dispatch.go)).

A campaign is a from number, a sequence, calling hours and a concurrency. A contact is a number
in E.164, optional `instructions`, `variables` and a `time_zone`. An attempt is one dial of one
contact, kept so a retry never overwrites what the last call did.

### States

A campaign is `draft` until launched. `POST .../launch` needs `consent: true`, which is recorded
as `consent_at`. It goes to `scheduled` when `start_at` is in the future, else `running`. The
runner moves `scheduled` to `running` once `start_at` passes, and `running` to `completed` once
nobody is left to ring. `paused` can be resumed. `stopped` is for good.

Once launched, only the name, the from number and the concurrency can change. The sequence and
hours are what the customer consented to ring people with. Only a `draft`, `stopped` or
`completed` campaign can be deleted; its calls stay.

A contact is `enrolled` (waiting for `next_attempt_at`), `in_progress` (claimed, a call
happening), `awaiting_outcome` (answered, transcript not read yet), `completed`, `failed`,
`rejected` or `stopped`. The API shows `awaiting_outcome` as `in_progress`. Contacts who opted out
of voice are `rejected` at launch, without a call.

### The sequence

A sequence has exactly one step, `step_1`, for now. A step names an agent config, a `goal`, 1 to
10 `attempts` with a delay in minutes before each, and the `outcomes` the call can reach. Each
outcome ends `successful` or `unsuccessful`. `other` and `no_response` have routes of their own,
and `do_not_call` is always offered. `double_dial` rings again at once when the first dial of an
attempt is not answered.

A new draft gets four attempts over twenty days, double dial on, and two outcomes (`success`,
`not_interested`).

### Calling hours

Windows per weekday on a 15-minute step, plus `excluded_dates`. The zone is either `fixed`, or
`contact`: the contact's own `time_zone`, else one guessed from the number's prefix, else
`fallback`. A new draft rings weekdays 09:00 to 18:00 where the contact is. A contact with no
opening in the next 400 days fails with `no_calling_hours`.

### The dialer

Every router runs a `Runner`, and all of them share one queue in Postgres. Every 2 seconds a
tick:

1. starts the scheduled campaigns that are due,
2. takes over contacts whose lease ran out,
3. claims due contacts, up to each campaign's `concurrency`,
4. reads the outcomes of answered calls,
5. completes campaigns with nobody left.

`ClaimDueContacts` locks the campaign row with `SKIP LOCKED`, counts its `in_progress` contacts,
and leases the free slots to this router for 20 minutes. So concurrency holds across routers.
The least busy campaigns are served first, so one big campaign does not starve another. One
router holds at most 100 calls, and a campaign asks for at most 50.

Before dialing, the runner checks the hours in the contact's zone, the opt-out list, and that
the campaign is still `running`. The agent config is read at each dial, so an edit reaches the
next person rung. The session gets the config's instructions with `{{variable}}` filled from
the contact, the step's goal, any variables the text never mentions, the contact's own
instructions, and last a rule naming the outcomes that end the call. A call is cut off after 15
minutes.

### What a dial becomes

How a placed call ended comes from
[internal/phone/outcome](../../acceleration/internal/phone/outcome): `completed` once the
person's phone joined the Stream call, whatever the vendor said, else the vendor's `busy`,
`no_answer`, `failed`, `canceled`, or `unknown` for a vendor that does not report. See
[telephony](telephony.md#how-a-placed-call-ended).

- **Not answered, busy, failed.** The attempt is used. The next one is due at the first opening
  after its delay. After the last, the contact completes with `no_response`.
- **Answered.** The contact waits in `awaiting_outcome`. About 10 seconds later a router reads
  the transcript and asks a model (the `multilingual-high-accuracy` tier) which outcome the
  call reached. An empty transcript is read again every 30 seconds, 6 times in all. A call
  where only the agent spoke, or only a recording did, counts as not answered. One that fits no
  outcome is `other`. The contact completes, and `converted` is set when the route ends
  `successful`.
- **`do_not_call`.** Also writes an opt-out for the number on voice, so no campaign rings it again.
- **Answering machine.** A step's `voicemail` is `off`, `hang_up` or `leave_message`. When it is
  not `off`, the call asks the vendor to detect a machine (Twilio and Telnyx do). On a machine
  the agent says the message word for word, or nothing, and hangs up. The attempt is used, and
  no second dial follows.
- **Invalid number.** A number that cannot be possible is refused when added. One the vendor
  will not dial fails the contact with `invalid_number`.
- **10DLC refusal.** A limit that resets (`concurrent`, `per_day`, ...) defers the contact: 5
  minutes for `concurrent`, the next local day for the rest. Any other refusal fails it with the
  refusal's code. See [10DLC](10DLC.md).
- **Refusals every call would hit** pause the whole campaign with `paused_reason`:
  `telephony_suspended`, `insufficient_credits`, `config_missing`, `session_refused`, or
  `voicemail_needs_exact_words` for a speech-to-speech config asked to leave a message.

Parallel calls from one number go through [shared trunks](shared-trunks.md) for Twilio, Telnyx
and SIP trunks.

### Routers coming and going

A router shutting down claims nothing new, places no new call, and waits up to 15 minutes for
the calls it holds. A transcript reading cut off is handed back at once.

A router that dies leaves its contacts leased. Once the 20-minute lease runs out, another router
takes each over and settles it from its last dial in `campaign_attempts`. It never dials it again.
A contact claimed but never dialed goes back to `enrolled`.

### Moving to another region

`router replicate` copies all three tables. A campaign `running` or `scheduled` at the source
arrives `paused` and marked `move_paused`, so two deployments never ring the same list.
`replicate --resume` names those campaigns and leaves them paused. A person resumes each once
the source's calls are over.

### Existing campaigns

[The migration](../../acceleration/migrations/20261218110000_campaign_dialer.sql) gives older
campaigns one step with one attempt and hours open around the clock in UTC, so they ring as
before. `finished` becomes `completed`. A number added twice keeps its first row; the rest are
`rejected` with reason `duplicate`.

## API

```
GET    /v1/agents/campaigns                                  list, newest first
POST   /v1/agents/campaigns                                  make a draft
GET    /v1/agents/campaigns/{id}                             one, contacts counted by stage
PATCH  /v1/agents/campaigns/{id}                             change settings
DELETE /v1/agents/campaigns/{id}                             delete it and its contacts
POST   /v1/agents/campaigns/{id}/launch                      launch a draft, with consent
POST   /v1/agents/campaigns/{id}/pause                       ring nobody new
POST   /v1/agents/campaigns/{id}/resume                      ring again
POST   /v1/agents/campaigns/{id}/stop                        end for good
POST   /v1/agents/campaigns/{id}/duplicate                   a draft with the same settings
GET    /v1/agents/campaigns/{id}/analytics                   dialed, connected, converted, per day
GET    /v1/agents/campaigns/{id}/contacts                    contacts, by stage or number
POST   /v1/agents/campaigns/{id}/contacts                    add up to 1,000 to a draft
DELETE /v1/agents/campaigns/{id}/contacts                    remove every contact of a draft
GET    /v1/agents/campaigns/{id}/contacts/{contact_id}       one contact, every dial made to them
POST   /v1/agents/campaigns/{id}/contacts/{contact_id}/stop  ring this one no more
```

All are server-side only. Pausing or stopping never hangs up a call in progress. A campaign
holds at most 100,000 contacts.

## Not done

- **One step only.** A route to another step is refused.
- **`auto_enroll` is reserved.** Nothing creates one, so contacts are only added to a draft.
- **Machine detection at two vendors.** Elsewhere no message is left. The transcript still
  tells a recording from a person, so the attempt counts as not answered.
- **A leg still ringing when its session fails is not hung up at the vendor.** Only the Stream
  call is ended.
- **The lease is never renewed.** That is safe only because a call is cut off at 15 minutes.
