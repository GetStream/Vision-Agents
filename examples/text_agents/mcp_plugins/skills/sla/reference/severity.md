# Mapping a vendor's severity to P1-P4

The script only knows P1 to P4. Map anything else first.

| Source | Value | Severity |
| --- | --- | --- |
| Sentry level | `fatal` | P1 |
| Sentry level | `error` | P2 |
| Sentry level | `warning` | P3 |
| Sentry level | `info`, `debug` | P4 |
| Linear priority | Urgent (1) | P1 |
| Linear priority | High (2) | P2 |
| Linear priority | Medium (3) | P3 |
| Linear priority | Low (4), No priority (0) | P4 |
| GitHub label | `sev1` ... `sev4` | P1 ... P4 |

A Sentry issue escalates by one level (P2 becomes P1, and so on) when it affects more than
1,000 users. Nothing escalates past P1.
