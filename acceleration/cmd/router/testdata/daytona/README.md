# Daytona analytics responses

These bodies are built from the OpenAPI document of Daytona's analytics API (v0.4.0, as
`github.com/daytona/clients/analytics-api-client-go` ships it, read 2026-10-23), not recorded
from real calls: the fields and their names follow the documented response of each endpoint,
and the values are made up to be easy to check.

- `usage_sandbox.json`: `GET /organization/{organizationId}/usage/sandbox?from&to`
  (`models.SandboxUsage`), one day of the organization's sandboxes, each a case
  `TestSandboxUsageIsReconciled` checks.
- `periods_sandbox-b.json`: `GET /organization/{organizationId}/sandbox/{sandboxId}/usage?from&to`
  (`models.UsagePeriod`), half an hour running at Daytona's default size, then an hour stopped.

The document types every quantity as an integer and does not say what `cpu`, `ramGB` and
`diskGB` of a period are; they are read here as the size held over the period, so a period's
vCPU seconds are its `cpu` times its length. Both are to be confirmed against real responses.
