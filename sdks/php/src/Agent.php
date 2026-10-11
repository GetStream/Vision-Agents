<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Folder\Declaration;
use GetStream\VisionAgents\Folder\Document;
use GetStream\VisionAgents\Folder\KnowledgeUrl;
use GetStream\VisionAgents\Folder\Simulation;
use GetStream\VisionAgents\Generated\AgentConfig;
use GetStream\VisionAgents\Generated\AgentMode;
use GetStream\VisionAgents\Generated\CreateSessionRequest;
use GetStream\VisionAgents\Generated\Harness as HarnessName;
use GetStream\VisionAgents\Generated\ModelOverwrites;
use GetStream\VisionAgents\Generated\PlaceCallRequest;
use GetStream\VisionAgents\Generated\PlacedCall;
use GetStream\VisionAgents\Generated\Sandbox;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Generated\SessionMemory;
use GetStream\VisionAgents\Generated\SessionPhone;
use GetStream\VisionAgents\Generated\SyncAgentRequest;
use GetStream\VisionAgents\Generated\SyncAgentResult;
use GetStream\VisionAgents\Inbound\InboundCall;
use GetStream\VisionAgents\Inbound\InboundMessage;

/**
 * A configured agent, before and between the calls it holds.
 *
 * An agent here is configuration and function calling. The conversation itself, joining the
 * call, hearing the caller, answering and speaking, happens in the backend.
 *
 *     $agent = new Agent(config: 'simple_voice_ai', costTracking: ['env' => 'production'], memoryFilter: ['user_id' => '123']);
 *     $session = $agent->join();
 *     $session->responses->create('greet the user in one short sentence');
 *
 * What is written in code wins over what a folder says. A directory is a starting point.
 */
final class Agent
{
    /** The memory filter key naming who the memories are about. Everything else narrows recall. */
    public const string USER_KEY = 'user_id';

    public readonly string $name;
    public readonly string $instructions;
    public readonly string $guardrail;
    public readonly string $userId;
    public readonly Tools $tools;
    public readonly Client $client;
    public readonly ?Folder $folder;
    private readonly ?Harness $harness;
    private ?Edge $edge;
    /** `config` resolved to the id the backend wants, looked up once. */
    private ?string $configId = null;

    /**
     * @param string $config a stored agent config to start from, by name or id; it is also the
     *     agent's name when nothing else names it
     * @param Folder|string|null $folder an agent directory, or the path to one
     * @param array<string, string> $costTracking labels every request the session makes, so spend
     *     can be attributed to whatever they mean to you
     * @param array<string, string|int> $memoryFilter who the memories are about, under `user_id`,
     *     and what narrows recall
     * @param ?Sandbox $sandbox where code the agent writes gets run
     * @param string $userId who the agent joins a call as; derived from the name when empty
     */
    public function __construct(
        public readonly string $config = '',
        string $name = '',
        string $instructions = '',
        string $guardrail = '',
        Folder|string|null $folder = null,
        public readonly Pipeline $pipeline = new Pipeline(),
        ?Harness $harness = null,
        public readonly ?Sandbox $sandbox = null,
        public readonly array $costTracking = [],
        public readonly array $memoryFilter = [],
        string $userId = '',
        ?Tools $tools = null,
        ?Client $client = null,
        ?Edge $edge = null,
    ) {
        $this->folder = is_string($folder) ? Folder::load($folder) : $folder;
        $this->name = $name !== '' ? $name : ($this->folder->name ?? $config);
        if ($this->name === '') {
            throw new ConfigurationException('an agent needs a name, a config or a folder');
        }
        $this->instructions = $instructions !== '' ? $instructions : ($this->folder->instructions ?? '');
        $this->guardrail = $guardrail !== '' ? $guardrail : ($this->folder->guardrail ?? '');

        // A directory's skills are folded in here, so everything downstream reads one harness.
        $fromFolder = $this->folder->skills ?? [];
        if ($fromFolder !== [] && ($harness === null || $harness->skills === [])) {
            $harness = ($harness ?? new Harness())->withSkills($fromFolder);
        }
        $this->harness = $harness;

        $this->userId = $userId !== '' ? $userId : self::userIdOf($this->name);
        $this->tools = $tools ?? new Tools();
        $this->client = $client ?? new Client();
        $this->edge = $edge;
    }

    /**
     * Holds a conversation on the session's own call, `agent:<session id>`.
     *
     * It returns once the backend is in the call, so an agent that has joined is one that is
     * already listening.
     */
    public function join(): Session
    {
        return $this->open(startVoice: true);
    }

    /**
     * Holds the conversation in writing rather than on a call. Everything between hearing a
     * question and answering it is unchanged: the same instructions, skills and knowledge. It is
     * kept in Stream Chat unless incognito. `Session::startVoice` puts the agent on its call.
     *
     * @param string $agentId the conversation being answered, which names the channel replies
     *     are written into; a worker answering several conversations has to set it
     * @param array<string, mixed>|null $custom
     */
    public function chat(
        string $agentId = '',
        ?string $title = null,
        ?string $description = null,
        ?string $projectId = null,
        ?array $custom = null,
        ?bool $incognito = null,
        ?ModelOverwrites $modelOverwrites = null,
    ): Session {
        return $this->open(
            agentId: $agentId === '' ? null : $agentId,
            title: $title,
            description: $description,
            projectId: $projectId,
            custom: $custom,
            incognito: $incognito,
            modelOverwrites: $modelOverwrites,
        );
    }

    /**
     * Carries on a conversation held in writing, by the id of the session it was held in.
     * `watch()` it to answer its tool calls with this agent's.
     */
    public function resume(string $id): Session
    {
        return new Session($this->client, SessionRow::fromArray(Json::asObject($this->client->get('/v1/agents/sessions/{id}', ['id' => $id]))), $this);
    }

    /**
     * Answers a call that arrived through dispatch, on the call the router routed the caller
     * into, which is named for the session opened here. The number they reached is carried in,
     * which is what lets the agent transfer.
     */
    public function answer(InboundCall $call): Session
    {
        if ($call->sessionId === '') {
            throw new ConfigurationException('the call names no session; attach its number again');
        }
        return $this->open(
            id: $call->sessionId,
            startVoice: true,
            phone: $call->calledNumber === '' ? null : new SessionPhone(number: $call->calledNumber),
        );
    }

    /**
     * Answers a message written to an agent that is not running, in the channel its agent id
     * names.
     */
    public function reply(InboundMessage $message): Session
    {
        return $this->chat(agentId: $message->agentId);
    }

    /**
     * Rings somebody and holds the conversation when they answer.
     *
     * The call is placed before the agent joins, because placing it pins its own routing rule to
     * the call of the session it names; attaching the number first would be a second rule for
     * the same number. The agent is told it is navigating, so recordings are let finish and
     * menus are answered.
     */
    public function outboundCall(string $from, string $to): Session
    {
        if ($from === '' || $to === '') {
            throw new ConfigurationException('a call needs a number to ring from and one to ring');
        }
        $request = new PlaceCallRequest(
            from: $from,
            to: $to,
            tags: $this->costTracking === [] ? null : $this->costTracking,
        );
        $placed = PlacedCall::fromArray(Json::asObject($this->client->post('/v1/phone/calls', body: $request->toArray())));
        if ($placed->sessionId === null || $placed->sessionId === '') {
            throw new RouterException(201, 'POST /v1/phone/calls', 'the router placed the call for no session');
        }
        return $this->open(
            id: $placed->sessionId,
            startVoice: true,
            navigating: true,
            phone: new SessionPhone(number: $from, vendorCallId: $placed->vendorCallId),
        );
    }

    /**
     * A link a person can open to join the session's call from a browser and hear the agent.
     * They join as a listener of their own, so opening it twice puts two people in the call.
     */
    public function monitorUrl(Session $session): string
    {
        if (!$session->voiceStarted()) {
            throw new ConfigurationException('a conversation held in writing has no call to watch');
        }
        return $this->edge()->monitorUrl($session->call(), 'monitor-' . $session->id(), 'Monitor');
    }

    /**
     * The agent's knowledge base, named after the agent.
     */
    public function knowledge(): Knowledge
    {
        return new Knowledge($this->client, $this->name);
    }

    /**
     * Stores the agent in the backend: instructions, guardrail, skills, knowledge, and the
     * models it was declared with, in one request.
     *
     * The request carries a fingerprint, and a folder records it in `.agent_sync`, so syncing
     * on every startup asks for nothing more than the stored config when nothing changed. A
     * setting left out leaves whatever is stored, so a model chosen in the dashboard survives.
     *
     * Server side only: how an agent is configured is not a device's to rewrite.
     */
    public function sync(): SyncAgentResult
    {
        $folder = $this->folder;
        $declared = $folder->settings ?? new Declaration();
        $skills = $this->harness->skills ?? [];
        $knowledge = $folder->knowledge ?? [];
        $pages = $folder->knowledgeUrls ?? [];
        $simulations = $folder?->simulations;
        $subagent = $this->harness !== null && $this->harness->subagent !== '' ? $this->harness->subagent : '';
        $harnessName = $this->harness === null ? '' : ($this->harness->name instanceof HarnessName ? $this->harness->name->value : $this->harness->name);
        $pipeline = $this->pipeline;

        $hash = Folder::fingerprint($folder->declaration ?? '', $this->instructions, $this->guardrail, $skills, $knowledge, $pages, $simulations);
        // What the code set is part of the fingerprint too, so changing it syncs again. With only
        // a harness, a subagent and labels set this is the Go SDK's fingerprint exactly.
        $coded = self::described([$pipeline->llm, $pipeline->stt, $pipeline->tts, $pipeline->sts, $pipeline->voice, $pipeline->greeting, $this->sandbox?->value]);
        if ($harnessName !== '' || $subagent !== '' || $this->costTracking !== [] || $coded !== '') {
            $hash = Folder::fingerprint($hash, $harnessName . $subagent, self::sprint($this->costTracking) . $coded);
        }

        if ($folder !== null && $folder->stamp() === $hash) {
            // A config deleted since the stamp was written is synced again rather than trusted.
            $stored = self::storedConfig($this->client, $this->name);
            if ($stored !== null) {
                return new SyncAgentResult(config: $stored, unchanged: true);
            }
        }

        $tags = [...$declared->tags, ...$this->costTracking];
        $body = new SyncAgentRequest(
            name: $this->name,
            hash: $hash,
            instructions: self::set($this->instructions),
            guardrail: self::set($this->guardrail),
            skills: $skills === [] ? null : array_map(static fn (Skill $skill) => $skill->toSync(), $skills),
            knowledge: $knowledge === [] ? null : array_map(static fn (Document $document) => $document->toKnowledge(), $knowledge),
            knowledgeUrls: $pages === [] ? null : array_map(static fn (KnowledgeUrl $page) => $page->toDeclaration(), $pages),
            mode: $declared->mode === '' ? null : (AgentMode::tryFrom($declared->mode) ?? $declared->mode),
            stt: $pipeline->stt ?? self::set($declared->stt),
            tts: $pipeline->tts ?? self::set($declared->tts),
            sts: $pipeline->sts ?? $declared->sts,
            voice: $pipeline->voice ?? self::set($declared->voice),
            llm: $pipeline->llm ?? self::set($declared->llm),
            video: $pipeline->video ?? $declared->video,
            harness: self::harness($harnessName) ?? self::harness($declared->harness),
            subagent: self::set($subagent) ?? self::set($declared->subagent),
            search: self::set($declared->search),
            greeting: $pipeline->greeting() ?? $declared->greeting,
            plugins: $declared->plugins === [] ? null : $declared->plugins,
            keyterms: $pipeline->keyterms ?? ($declared->keyterms === [] ? null : $declared->keyterms),
            sandbox: $this->sandbox ?? ($declared->sandbox === '' ? null : (Sandbox::tryFrom($declared->sandbox) ?? $declared->sandbox)),
            tags: $tags === [] ? null : $tags,
            dispatch: $declared->dispatch,
            simulations: $simulations === null ? null : array_map(static fn (Simulation $simulation) => $simulation->toDeclaration(), $simulations),
        );
        $result = SyncAgentResult::fromArray(Json::asObject($this->client->post('/v1/agents/sync', body: $body->toArray())));
        $folder?->writeStamp($hash);
        return $result;
    }

    /**
     * The config stored under a name, or null when there is none.
     */
    public static function storedConfig(Client $client, string $name): ?AgentConfig
    {
        // Narrowed to the name rather than read whole and filtered here: names are unique per
        // customer, so this asks the router the question instead of downloading every config.
        foreach (Json::objects(['rows' => $client->get('/v1/agents/configs', query: ['name' => $name])], 'rows') as $row) {
            $config = AgentConfig::fromArray($row);
            if ($config->name === $name) {
                return $config;
            }
        }
        return null;
    }

    /**
     * Turns a name into something a call can be joined under.
     */
    public static function userIdOf(string $name): string
    {
        $id = trim((string) preg_replace('/[^a-z0-9_-]/', '-', strtolower($name)), '-');
        return $id === '' ? 'vision-agent' : $id;
    }

    /**
     * Renders the agent's configuration into a session and opens it. Only what was set is
     * sent, so the config or the router decides the rest.
     *
     * The parameters are what is particular to this way of opening one.
     *
     * @param array<string, mixed>|null $custom
     */
    private function open(
        ?string $id = null,
        ?bool $startVoice = null,
        ?SessionPhone $phone = null,
        ?bool $navigating = null,
        ?string $agentId = null,
        ?string $title = null,
        ?string $description = null,
        ?string $projectId = null,
        ?array $custom = null,
        ?bool $incognito = null,
        ?ModelOverwrites $modelOverwrites = null,
    ): Session {
        $pipeline = $this->pipeline;
        $tools = $this->tools->declared();

        // The harness, the sandbox and the skills are the config's, written by `sync`, never
        // the session's.
        $request = new CreateSessionRequest(
            id: $id,
            startVoice: $startVoice,
            configId: $this->config === '' ? null : $this->resolveConfig(),
            incognito: $incognito,
            title: $title,
            description: $description,
            projectId: $projectId,
            custom: $custom,
            modelOverwrites: $modelOverwrites,
            userId: $this->userId,
            userName: $this->name,
            agentId: $agentId ?? $this->userId,
            greeting: $pipeline->greeting(),
            navigating: $navigating,
            llm: $pipeline->llm,
            stt: $pipeline->stt,
            tts: $pipeline->tts,
            sts: $pipeline->sts,
            voice: $pipeline->voice,
            languages: $pipeline->language === null ? null : [$pipeline->language],
            keyterms: $pipeline->keyterms,
            maxTokens: $pipeline->maxTokens,
            backchannel: $pipeline->backchannel,
            tools: $tools === [] ? null : $tools,
            toolTimeoutMs: $pipeline->toolTimeoutMs,
            tags: $this->costTracking === [] ? null : $this->costTracking,
            memory: $this->memory(),
            phone: $phone,
            video: $pipeline->video,
        );
        $created = $this->client->post('/v1/agents/sessions', body: $request->toArray());
        return new Session($this->client, SessionRow::fromArray(Json::asObject($created)), $this);
    }
    
    private function memory(): ?SessionMemory
    {
        if ($this->memoryFilter === []) {
            return null;
        }
        $userId = null;
        $narrowing = [];
        foreach ($this->memoryFilter as $key => $value) {
            if ($key === self::USER_KEY) {
                $userId = (string) $value;
                continue;
            }
            $narrowing[$key] = (string) $value;
        }
        return new SessionMemory(filter: $narrowing === [] ? null : $narrowing, userId: $userId);
    }

    /**
     * A config name matching nothing stored is passed through, since it is then either an id or
     * a mistake the router can report better than a guess here.
     */
    private function resolveConfig(): string
    {
        return $this->configId ??= self::storedConfig($this->client, $this->config)->id ?? $this->config;
    }

    private function edge(): Edge
    {
        // Built lazily, so an agent that only ever chats needs no Stream credentials.
        return $this->edge ??= new Edge();
    }

    private static function set(string $value): ?string
    {
        return $value === '' ? null : $value;
    }

    private static function harness(string $name): HarnessName|string|null
    {
        return $name === '' ? null : (HarnessName::tryFrom($name) ?? $name);
    }

    /**
     * A map the way Go's fmt.Sprint prints one, keys sorted, which is what the Go SDK puts into
     * the fingerprint for the labels.
     *
     * @param array<string, string> $labels
     */
    private static function sprint(array $labels): string
    {
        ksort($labels, SORT_STRING);
        $pairs = [];
        foreach ($labels as $key => $value) {
            $pairs[] = "{$key}:{$value}";
        }
        return 'map[' . implode(' ', $pairs) . ']';
    }

    /**
     * @param list<?string> $values
     */
    private static function described(array $values): string
    {
        $set = array_filter($values, static fn (?string $value): bool => $value !== null);
        return $set === [] ? '' : "\n" . implode("\n", array_map(static fn (?string $value): string => (string) $value, $values));
    }
}
