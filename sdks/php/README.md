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

// Your own server, against a hosted router.
$client = new Client(new Backend(apiKey: $key, apiSecret: $secret, authenticate: true));
```

Every argument falls back to the environment: `STREAM_ACCELERATION_URL`,
`STREAM_ACCELERATION_CUSTOMER_ID`, `STREAM_API_KEY`, `STREAM_API_SECRET`,
`STREAM_ACCELERATION_AUTHENTICATE`. Pass the client to anything that talks to the router:
`new Agent(..., client: $client)`.

## Conversations, responses, rewind and fork

```php
use GetStream\VisionAgents\Generated\ForkSessionRequest;

$session = $agent->chat();
$first = $session->responses->create('Name a colour.');
$second = $session->responses->create('Now a darker one.');

foreach ($second->items->unwind() as $item) {
    echo $item->kind, "\n";
}

$session->responses->rewind($first);                       // forget everything after it
$fork = $session->fork(new ForkSessionRequest(responseId: $first->id()));
```

A response is created running and finishes on its own. Rewind takes a response `id`, not a
`turn_id`. A conversation kept with `persist: true` is refused a rewind with a 400: fork it
instead.

Past conversations, by agent:

```php
$rows = $client->agent('support')->sessions->query(state: 'closed', custom: ['plan' => 'pro']);
```

## An agent from a directory

```
support/
  agent.yaml          name, models, mode, keyterms, video, ...
  instructions.md
  guardrail.md
  skills/refunds.md   frontmatter: description, deadline, capture_video
  knowledge/*.md
  knowledge/urls.yaml pages to keep the knowledge base filled from
```

```php
$agent = new Agent(folder: __DIR__ . '/support');
$agent->sync();
```

`sync` is one request carrying everything, knowledge URLs included. It writes `.agent_sync`
with the fingerprint, and while nothing changes a later sync reads the stored config instead
of sending it again. The fingerprint is the Go SDK's, so either can sync the same directory.
An unknown key in `agent.yaml` is refused rather than ignored.

```php
$agent->knowledge()->addUrl('https://example.com/pricing', title: 'Pricing');
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
$watch->respond('Weather in Paris?');
foreach ($watch as $event) {
    if ($event->kind === 'responded') {
        echo $event->text(), "\n";
        break;
    }
}
```

A tool runs while the session is watched, in its own fiber. What it throws is sent to the
model as the tool's error.

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
    $dispatch->getOrCreateAgent($message, fn () => new Agent(config: 'support'));
});

$dispatch->run();
```

A handler that throws rejects the call. SIGINT and SIGTERM stop it where pcntl is loaded;
work still running is waited for.

## Guests

```php
$guest = $client->guestUser(name: 'Ada');         // hand $guest->token to the browser
$client->claimGuestUser($guest, $accountId);      // once they sign up
```

## Routing without an agent

```php
use GetStream\VisionAgents\Generated\{SttOptions, TtsOptions};
use GetStream\VisionAgents\Router;

$router = new Router('healthcare', tags: ['team' => 'clinical']);

$answer = $router->search('perioperative antibiotic guidance');
$transcript = $router->stt->recording('https://example.com/visit.mp3', new SttOptions(diarize: true));
$speech = $router->tts->recording('Chapter one.', new TtsOptions(voice: 'custom:reader'));

$stt = $router->stt->realtime();          // also tts, llm, sts
$stt->sendAudio($pcm16);
foreach ($stt->frames() as $frame) { /* transcript frames */ }
```

## Every endpoint

`Client` has one method per HTTP method. Request and response shapes are generated into
`GetStream\VisionAgents\Generated` from `acceleration/api/openapi.yaml`.

```php
$configs = $client->get('/v1/agents/configs', query: ['name' => 'support']);
$client->delete('/v1/agents/sessions/{id}', ['id' => $id]);
```

A failure raises `RouterException` with the status (0 when the router was never reached),
the operation, and what the router said.

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
