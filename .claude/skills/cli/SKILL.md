---
name: cli
description: How the Stream CLI (getstream) reaches the agents backend, and how to point it at a local router or the hosted one. Read before running getstream agents commands or debugging why one fails.
---

# Stream CLI

The CLI is `getstream`, source in `workspace/cli` (GetStream/cli). Install or update it with:

```sh
curl -fsSL https://getstream.io/cli/beta/install.sh | bash
```

## Which router it talks to

`agents_base` in `~/.stream/config.yaml` decides where every `getstream agents` command goes:

```yaml
agents_base: http://localhost:8080 # the local router
# agents_base: https://appkey@accelerate.gcp.stream-io-api.com # the hosted one
```

- A bare URL names the app by its id in trusted headers (`X-Customer-Id` and `X-Stream-App-Id`), which a local router with no auth reads. The customer id is the app id from the project's `.stream/creds.yaml`.
- `appkey@` before the host signs as the app with its key and secret, the way the hosted proxy wants it.
- Left out, it is the hosted router.

There is no flag or env var for it. To run one command elsewhere without editing your config, give it a home of its own:

```sh
mkdir -p /tmp/clihome && cp -R ~/.stream /tmp/clihome/.stream
echo 'agents_base: https://appkey@accelerate.gcp.stream-io-api.com' > /tmp/clihome/.stream/config.yaml
HOME=/tmp/clihome getstream agents test my-agent
```

The hosted router is deployed less often than `accelerate` moves, so test new router behaviour against the local one (rebuild it first, see `AGENTS.md`). The SDKs pick their router separately, from `STREAM_ACCELERATION_URL` and `STREAM_ACCELERATION_CUSTOMER_ID`; set the customer id to the app id to reach the same agents the CLI does locally.

## Projects

`getstream init` links the current directory to an app by writing `.stream/creds.yaml`. Without a terminal it writes `.stream/init-*.yaml` instead: uncomment one `app_id` and run `getstream init --command <file>`.

## Debugging

- A non-2xx answer prints only the status, even with `--verbose`. To read the error message, repeat the request with an SDK, which surfaces the router's `{"error": {"message"}}`.
- `STREAM_API_KEY` and `STREAM_API_SECRET` in the environment override the project's credentials.
- The CLI checks `agent.yaml` keys itself before syncing, against its own list, so a key the router already accepts can still be refused by an older CLI.
