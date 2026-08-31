# Dashboard

Watches calls the acceleration router is holding, and configures what it holds them with.

It is a Next.js app that talks to the router straight from the browser: there is no server
of its own, no database and nothing cached between the two. What you see is what the router
says, and a call that is still running says it over the same WebSocket the SDK uses.

## Running it

```bash
npm install
npm run dev
```

Two settings, both public because both reach the browser either way:

| Variable                  | Default                 | What it is                            |
| ------------------------- | ----------------------- | ------------------------------------- |
| `NEXT_PUBLIC_ROUTER_URL`  | `http://localhost:8080` | Where the router is                   |
| `NEXT_PUBLIC_CUSTOMER_ID` | empty                   | Who the dashboard talks to it as       |

Put them in `dashboard/.env.local`. The router has to allow the dashboard's origin, so start
it with `ROUTER_CORS_ORIGINS=http://localhost:3000`.

## Pages

- `/` - the last calls and what they cost
- `/calls/{id}` - one call: who was talking when, what the agent heard, every judgement it
  made and what each turn took. Live while the call runs, from storage once it has ended.
  A call that is still running can also be joined from here, to talk to the agent: the
  router mints the token, so the secret stays where the rest of them are and this page
  still has no server behind it. You join with the camera off and the microphone on, and
  which microphone, speaker and camera is yours to pick.
- `/agents` - agent configs: instructions, models, skills, knowledge
- `/voices` - voices of your own, their recordings and how each provider is getting on with
  them
- `/telephony` - the numbers you hold, and buying and attaching new ones

## Types

`src/lib/api.d.ts` is generated from the router's spec and should never be edited:

```bash
npm run types
```

CI regenerates it and fails on a diff, the same way it does for the Go and Python clients.
