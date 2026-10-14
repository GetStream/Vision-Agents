# Why the targets are what they are

The numbers live in `scripts/policy.json`; this is the reasoning behind them.

- **P1** is an outage or data loss for many customers. Somebody is paged around the clock,
  so its clock never stops: 15 minutes to acknowledge and 4 hours to resolve or mitigate.
- **P2** is a major feature broken for many, or an outage for one large customer. Also
  around the clock: 1 hour to acknowledge, 1 day to resolve.
- **P3** is a bug with a workaround. It is handled in business hours only, 09:00 to 17:00
  UTC, Monday to Friday: 1 business day to acknowledge, 5 to resolve.
- **P4** is cosmetic or a question. Business hours too: 2 business days to acknowledge, 20
  to resolve.

A business-hours clock pauses outside those hours, so a P3 opened on Friday at 16:30 has
used 30 minutes of its first business day by the time the office closes, and carries on on
Monday at 09:00.
