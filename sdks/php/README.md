# getstream/vision-agents

The PHP client for the Stream acceleration backend, for the server side of a PHP app: open
conversations, run agents from a directory, place and answer calls, and route speech, text
and search.

Nothing here does inference or touches media. The backend joins the call, hears the caller,
answers and speaks. What stays here is configuration and function calling.

```bash
composer require getstream/vision-agents
```

PHP 8.4 or newer. Any PSR-18 client works; Guzzle arrives with Stream's own PHP SDK, which
this uses to create calls. Sockets (watching a session, dispatch, realtime streams) need Amp:

```bash
composer require amphp/websocket-client amphp/http-client-psr7
```

## An agent on a call

```php
use GetStream\VisionAgents\Agent;

$agent = new Agent(
    config: 'simple_voice_ai',
    costTracking: ['env' => 'production'],
    memoryFilter: ['user_id' => '123'],
);

$session = $agent->join('hello');
$session->responses->create('Greet the user in one short sentence.');
echo $agent->monitorUrl($session), "\n";
```

`join` creates the Stream call with `STREAM_API_KEY` and `STREAM_API_SECRET` and returns once
the backend is in it. What is written in code wins over the stored config; what is left out
is the config's to decide.

## Who is calling

```php
use GetStream\VisionAgents\{Backend, Client};

// A router with nothing in front of it, on a laptop.
$client = new Client(new Backend(url: 'http://localhost:8080', customerId: 'examples'));

// Your own server, against Stream's hosted router: STREAM_API_KEY and STREAM_API_SECRET are read for you.
$client = new Client();
```

Every argument falls back to the environment: `STREAM_ACCELERATION_URL`,
`STREAM_ACCELERATION_CUSTOMER_ID`, `STREAM_API_KEY`, `STREAM_API_SECRET`,
`STREAM_ACCELERATION_AUTHENTICATE`, so `new Client()` is usually enough. With no URL the client
goes to Stream's hosted router, through its authenticating proxy. Pass the client to
anything that talks to the router: `new Agent(..., client: $client)`.

## Conversations, responses, rewind and fork

```php
use GetStream\VisionAgents\Generated\ForkSessionRequest;

$session = $agent->chat();
$first = $session->responses->create('Name a colour.');
$second = $session->responses->create('Now a darker one.');

foreach ($second->items->unwind() as $item) {
    echo $item->kind, "\n";
}

$fork = $session->fork(new ForkSessionRequest(responseId: $first->id()));  // carry on from the first
```

A response is created running and finishes on its own. A written conversation is kept in
Stream Chat unless it is opened with `incognito: true`. Neither can be rewound (the router
answers 400): fork at the response instead. `$session->responses->rewind()` is for a call, and
takes a response `id`, not a `turn_id`.

One method changes a session: title, description, custom labels, instructions, models and
voice, from the next turn, for this session only. An ended session can still be renamed by id:

```php
$row = $session->update(title: 'Pricing', llm: 'llm-thinking', thinking: 'high');
$client->agent('support')->sessions->update($sessionId, title: 'Pricing');
```

Past conversations, by agent, a page at a time:

```php
$sessions = $client->agent('support')->sessions;
$page = $sessions->query(userId: 'u1', state: 'live');
$next = $sessions->query(userId: 'u1', state: 'live', cursor: $page->nextCursor);
$found = $sessions->search('pricing');
```

`close()` stops a session and keeps what it recorded and remembered; `delete()` deletes it,
its turns and its memories. Memory can also be deleted on its own (server side only):

```php
$session->deleteMemories();                 // what this session learned
$client->memories->truncate('u1');          // everything remembered about one user
```

## An agent from a directory

```
support/
  agent.yaml          name, models, mode, speed, harness, keyterms, video, dispatch, ...
  instructions.md
  guardrail.md
  skills/refunds.md   frontmatter: description, deadline, capture_video
  knowledge/*.md
  knowledge/urls.yaml pages to keep the knowledge base filled from, each with an optional refresh_hours
  simulations/*.yaml  lists of simulations: name, scenario, assertion, variations, ...
```

```php
$agent = new Agent(folder: __DIR__ . '/support');
$agent->sync();
```

`sync` is one request carrying everything, knowledge URLs and simulations included. The
harness (`harness:`, the subagent, the sandbox and the skills) is written onto the config,
never onto a session. A `simulations/` directory makes the stored simulations exactly the ones
it declares, so an empty one deletes them; without the directory they are left alone. `sync`
writes `.agent_sync` with the fingerprint, and while nothing changes a later sync reads the
stored config instead of sending it again. The fingerprint is the Go SDK's, so either can sync
the same directory. An unknown key in `agent.yaml`, `urls.yaml` or a simulations file is
refused rather than ignored.

```php
$agent->knowledge()->addUrl('https://example.com/pricing', title: 'Pricing', refreshHours: 24);
```

To change part of a stored config without restating the rest (server side only):

```php
use GetStream\VisionAgents\Generated\AgentConfigPatch;

$client->agent('support')->updateConfig(new AgentConfigPatch(guardrail: 'Never promise a refund.'));
```

## Tools

```php
$agent->tools->register('get_weather', 'The weather in a city', [
    'type' => 'object',
    'properties' => ['city' => ['type' => 'string']],
    'required' => ['city'],
], fn (array $args): array => ['city' => $args['city'], 'sky' => 'clear']);

$session = $agent->chat();
$watch = $session->watch();
$session->responses->create('Weather in Paris?');
foreach ($watch as $event) {
    if ($event->kind === 'responded') {
        echo $event->text(), "\n";
        break;
    }
}
```

A tool runs while the session is watched, in its own fiber. What it throws is sent to the
model as the tool's error. `register` also takes `displayTitle:`, what the reply's tool call
shows, and `executor: 'client'` for a tool a person's device runs instead.

## Phone

```php
$session = $agent->outboundCall(from: '+15550001111', to: '+15552223333');
```

## Workers: calls and messages the router hands you

Run this as a long-lived CLI process, not in a web request.

```php
use GetStream\VisionAgents\Inbound\{InboundCall, InboundMessage};
use GetStream\VisionAgents\Worker\Dispatch;

$dispatch = new Dispatch(capacity: 4);

$dispatch->waitForCall(function (InboundCall $call): void {
    (new Agent(config: 'support'))->answer($call)->watch()->wait();
});

$dispatch->waitForMessage(function (InboundMessage $message) use ($dispatch): void {
    if ($message->sessionId !== '') {
        $dispatch->answer($message); // text written to a running session
        return;
    }
    $dispatch->getOrCreateAgent($message, fn () => new Agent(config: 'support'));
});

$dispatch->run();
```

Each call and message ends with a `done` frame to the router, carrying what the handler threw
if it threw. SIGINT and SIGTERM stop it where pcntl is loaded; work still running is waited for.

An agent whose `agent.yaml` says `dispatch: {text: enabled}` leaves what end users write to
the worker: the message arrives with `sessionId` and `commandId`, and `answer()` has the
model reply on that session, using the worker's server credential acting for the writer.

### Hosted tools

A worker can also run tools for every session opened under an agent id, including sessions
opened from a browser. Hosting alone is enough to `run()`.

```php
use GetStream\VisionAgents\Tools;

$tools = (new Tools())->register('get_weather', 'The weather in a city', [
    'type' => 'object',
    'properties' => ['city' => ['type' => 'string']],
], fn (array $args): string => weatherIn($args['city']));

$dispatch->host('my-agent', $tools, timeoutMs: 0); // 0 takes the router's default
$dispatch->run();
```

Each call runs in its own fiber. A refusal throws `HostingRefusedException`, and
`$dispatch->hosting` lists the agent ids the router accepted.

## Guests

```php
$guest = $client->guestUser(name: 'Ada');         // hand $guest->token to the browser
$client->claimGuestUser($guest, $accountId);      // once they sign up
```

## Routing without an agent

The router comes from the client. `healthcare` is a stored router config, and it holds the
target each modality answers with (`target: en-low-latency` under `stt:` in
`routers/healthcare/router.yaml`, or `configureStt`); a call only overrides per-call options
such as `diarize` or `keyterms`.

```php
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Generated\{SttOptions, TtsOptions};

$router = (new Client())->router('healthcare', tags: ['team' => 'clinical']);

$answer = $router->search('perioperative antibiotic guidance');
$transcript = $router->stt->recording('https://example.com/visit.mp3', new SttOptions(diarize: true));
$speech = $router->tts->recording('Chapter one.', new TtsOptions(voice: 'custom:reader'));

$stt = $router->stt->realtime();          // also tts, llm, sts
$stt->sendAudio($pcm16);
foreach ($stt->frames() as $frame) { /* transcript frames */ }
```

## Simulations

Conversations to put an agent through, judged at the end (server side only):

```php
use GetStream\VisionAgents\Generated\SimulationRequest;

$simulation = $client->simulations->create(new SimulationRequest(
    name: 'lunch order',
    configId: $config->id,
    scenario: 'Order a turkey club, then swap it for a veggie wrap.',
    assertion: 'The final order is one veggie wrap.',
));
$run = $client->simulations->run($simulation->id);
$run = $client->simulations->runs->get($run->id);   // until its state is no longer running
```

Also `get`, `list`, `update`, `delete`, and `runs->list`, `runs->cancel`.

## Shapes and failures

Request and response shapes are generated into `GetStream\VisionAgents\Generated` from
`acceleration/api/openapi.yaml`. A failure raises `RouterException` with the status (0 when
the router was never reached), the operation, and what the router said.

## Developing

Docker only, from the repo root:

```bash
docker run --rm -v "$PWD":/repo -v "$PWD/sdks/php/build/composer-cache":/tmp/composer-cache \
  -e COMPOSER_CACHE_DIR=/tmp/composer-cache -w /repo/sdks/php composer:2 install
docker run --rm -v "$PWD":/repo -w /repo/sdks/php php:8.4-cli php bin/generate.php ../../acceleration/api/openapi.yaml --check
docker run --rm -v "$PWD":/repo -w /repo/sdks/php php:8.4-cli vendor/bin/phpunit --testsuite unit
docker run --rm -v "$PWD":/repo -w /repo/sdks/php php:8.4-cli vendor/bin/phpstan analyse --memory-limit=1G
docker run --rm -v "$PWD":/repo -w /repo/sdks/php -e VISION_AGENTS_URL=http://host.docker.internal:8091 \
  php:8.4-cli vendor/bin/phpunit --testsuite live
```
