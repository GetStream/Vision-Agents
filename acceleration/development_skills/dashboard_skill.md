# Local acceleration and Volt dashboard

Use this when running acceleration, an acceleration-backed agent, or an example
with the Volt dashboard. Follow the example's own README for model requirements.
Use the existing conversation screen; do not redesign it.


## Layout

Sibling checkouts. Volt lives next to Vision Agents, not inside it:

```text
vision-agents/          # this repo (compose.yaml, .env)
volt-dashboard/         # https://github.com/GetStream/volt-dashboard.git
```

```bash
export VISION_CHECKOUT="$(git rev-parse --show-toplevel)"
export VOLT_CHECKOUT="${VOLT_CHECKOUT:-$(dirname "$VISION_CHECKOUT")/volt-dashboard}"
export VOLT_PORT="${VOLT_PORT:-3011}"
```

If the sibling is missing, locate an existing Volt checkout or clone it there.
Make sure you are on the branch ` ai-team/agent-dashboard` for Volt. Read Volt's `AGENTS.md` before
changing that repo.

**Org `1181507` / app `1257545`.** All Stream keys, configs, and URLs use this
pair. 

## Prerequisites

Docker running. `ssh-add -l` must show a key with access to
`GetStream/getstream-go-webrtc`. Node 22+ (honor Volt's pin if newer). Bun for
all Volt commands, not npm/yarn/pnpm:

```bash
test -f "$VOLT_CHECKOUT/package.json"
test -f "$VOLT_CHECKOUT/scripts/agents-dev-proxy.ts"
docker info && docker compose version && ssh-add -l && node --version
command -v bun >/dev/null || { curl -fsSL https://bun.com/install | bash; export PATH="$HOME/.bun/bin:$PATH"; }
```

Do not start duplicate services or stop someone else's running call.

## Credentials

Preserve existing `.env` / `.env.local`. Copy an example file only when the
destination is missing; merge, never overwrite.

- **Vision Agents** root `.env`: `STREAM_API_KEY` / `STREAM_API_SECRET`, plus only the provider keys the example needs. Never put these in
  browser-visible `VITE_*` variables.
- **Volt** `.env.local`: follow Volt's `.env.example` and README. Keep the remote
  `VITE_AMPERE_API`. Do **not** copy `.env.local.example` — it switches to local
  Ampere and breaks HTTPS.
- **Missing keys:** ask the user to set up dashboard API keys from 1Password,
  including org `1181507` / app `1257545`. Do not invent a 1Password item name or
  substitute another app's credentials. Do not print secret values.

## Host and Volt env

`/etc/hosts` must map `local.getstream.io` to `127.0.0.1`. Append if missing;
do not replace the file. If it already points elsewhere, fix that first.

```bash
grep -E '^[^#]*[[:space:]]local\.getstream\.io([[:space:]]|$)' /etc/hosts ||
  printf '\n127.0.0.1       local.getstream.io\n' | sudo tee -a /etc/hosts
```

Merge into Volt's `.env.local` (same port everywhere: env, CORS, URL):

```dotenv
HOST=local.getstream.io
PORT=3011
AGENTS_ROUTER_URL=http://localhost:8080
```

Leave `AGENTS_STATS_ROUTER_URL` unset unless a separate usage reader is already
configured. Pick a free port explicitly; do not let Vite silently move. Reuse
Volt's existing proxy (`scripts/agents-dev-proxy.ts`); do not add another.

## Router

From the Vision Agents root. Temporary override so the router accepts Volt's
HTTPS origin:

```bash
cd "$VISION_CHECKOUT"
export VOLT_COMPOSE_OVERRIDE="$(mktemp /tmp/volt-agents-compose.XXXXXX)"
cat > "$VOLT_COMPOSE_OVERRIDE" <<EOF
services:
  router:
    environment:
      ROUTER_AUTH_MODE: noauth
      ROUTER_CORS_ORIGINS: https://local.getstream.io:${VOLT_PORT}
EOF

docker compose -f compose.yaml -f "$VOLT_COMPOSE_OVERRIDE" up -d --build router
curl --fail --silent --show-error http://localhost:8080/v1/agents/configs \
  -H 'X-Customer-Id: 1257545'
```

Router **8080**, Postgres **55432**, Redis **56379**. Keep the override path for
later Compose commands. Rebuild after backend changes; restart after env changes
once calls end. Never `down -v`, drop a database, or reset migrations.

## Volt

```bash
cd "$VOLT_CHECKOUT"
bun install
bun run dev
```

Keep it running. Confirm HTTPS on the configured port (`bun preview` will not
work). Sign in at:

```text
https://local.getstream.io:<PORT>/organization/1181507/1257545/agents/
```

Reuse a matching saved config in app `1257545`; create one from the example if
needed. Get the ID from the dashboard or `GET /v1/agents/configs`. There is no
`/agents/<example-name>` route.

| Destination | After `/agents/` |
| --- | --- |
| Config | `agent/configurations/<CONFIG_ID>/` |
| New voice test | `testing/playground/?config_id=<CONFIG_ID>&mode=voice` |
| Existing voice test | `testing/playground/?config_id=<CONFIG_ID>&mode=voice&session_id=<SESSION_ID>` |
| Logs | `agent/logs/?config_id=<CONFIG_ID>&range=%2224%22&live=on` |
| Session | `session/explorer/<SESSION_ID>/` |

`mode=text` for text. URL-encode IDs. Open the link and report it to the user.

## Run

**Playground (preferred).** Open the filtered playground URL. Start the test
**once** and wait for the session ID — retrying creates a second agent. For
voice, grant mic permission and enable the microphone (it starts off). End the
session when done; leaving the call does not stop the agent.

**CLI session**, same router. Use a real saved `CONFIG_ID` and a fresh call ID:

```bash
curl --fail --silent --show-error http://localhost:8080/v1/agents/sessions \
  -H 'X-Customer-Id: 1257545' -H 'Content-Type: application/json' \
  --data "{\"config_id\":\"${CONFIG_ID}\",\"call_id\":\"volt-example-$(date +%s)\"}"
```

Open the returned `id` on the existing-session link; do not also start a
playground test. Text: `{"config_id":"<CONFIG_ID>","text":true}` and `mode=text`.
Stop with `DELETE /v1/agents/sessions/${SESSION_ID}`.

**Standalone Go agent** only if explicitly requested. See `acceleration/README.md`
and `cmd/agent/main.go`. Host processes need mapped ports, not Compose hostnames:
`ROUTER_POSTGRES_DSN=postgres://postgres:postgres@localhost:55432/model_router?sslmode=disable`
and `ROUTER_REDIS_ADDR=localhost:56379`. The router cannot manage this process.

**Python examples:** follow that example's README with `uv`. They do not appear
in Volt unless their acceleration provider and this org/app are configured.

## If it fails

- Router: `docker compose -f compose.yaml -f "$VOLT_COMPOSE_OVERRIDE" logs --tail=100 router`
- Empty agents/logs: wrong customer ID, config, or time range
- SSH build failure: forwarded agent / `getstream-go-webrtc` access
- Login: prompt for the 1Password dashboard env; keep remote Ampere
- Browser: `/etc/hosts`, cert trust, Vite port, matching CORS origin
