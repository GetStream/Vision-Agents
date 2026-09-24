<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\RouterConfig;
use GetStream\VisionAgents\Generated\RouterConfigRequest;
use GetStream\VisionAgents\Generated\SearchAnswer;
use GetStream\VisionAgents\Generated\SearchOptions;
use GetStream\VisionAgents\Generated\SearchRequest;
use GetStream\VisionAgents\Generated\StsOptions;
use GetStream\VisionAgents\Generated\SttOptions;
use GetStream\VisionAgents\Generated\TtsOptions;
use GetStream\VisionAgents\Router\Completions;
use GetStream\VisionAgents\Router\SpeechToSpeech;
use GetStream\VisionAgents\Router\SpeechToText;
use GetStream\VisionAgents\Router\TextToSpeech;

/**
 * Everything the backend routes, configured once, for a pipeline that is not an agent.
 *
 *     $router = new Router('healthcare', tags: ['team' => 'clinical']);
 *     $answer = $router->search('perioperative antibiotic guidance');
 *     $transcript = $router->stt->recording('https://example.com/visit.mp3');
 *
 * Everything in the named config is a default, and every option on a call overrides one field
 * of it.
 */
final readonly class Router
{
    public SpeechToText $stt;
    public TextToSpeech $tts;
    public Completions $llm;
    public SpeechToSpeech $sts;
    public Client $client;

    /**
     * @param string $config a stored router config, by name or id; without one every call says
     *     what it wants for itself
     * @param array<string, string> $tags cost labels carried onto everything routed here
     */
    public function __construct(public string $config = '', public array $tags = [], ?Client $client = null)
    {
        $this->client = $client ?? new Client();
        $this->stt = new SpeechToText($this);
        $this->tts = new TextToSpeech($this);
        $this->llm = new Completions($this);
        $this->sts = new SpeechToSpeech($this);
    }

    /**
     * One question, one answer. No socket, because nothing arrives in pieces.
     */
    public function search(string $query, ?SearchOptions $options = null): SearchAnswer
    {
        $body = new SearchRequest($query, $this->configId(), $options, $this->labels());
        return SearchAnswer::fromArray(Json::asObject($this->client->post('/v1/search', body: $body->toArray())));
    }

    /**
     * Stores how this router transcribes. The other modalities are carried forward as stored,
     * since configuring how something is heard is not a statement about how it speaks.
     */
    public function configureStt(SttOptions $options): RouterConfig
    {
        return $this->store(static fn (string $name, ?RouterConfig $held) => new RouterConfigRequest($name, $options, $held?->tts, $held?->llm, $held?->sts, $held?->search));
    }

    /**
     * Stores how this router speaks, leaving the rest as stored.
     */
    public function configureTts(TtsOptions $options): RouterConfig
    {
        return $this->store(static fn (string $name, ?RouterConfig $held) => new RouterConfigRequest($name, $held?->stt, $options, $held?->llm, $held?->sts, $held?->search));
    }

    /**
     * Stores how this router holds a conversation with one native audio model.
     */
    public function configureSts(StsOptions $options): RouterConfig
    {
        return $this->store(static fn (string $name, ?RouterConfig $held) => new RouterConfigRequest($name, $held?->stt, $held?->tts, $held?->llm, $options, $held?->search));
    }

    /**
     * @internal
     */
    public function configId(): ?string
    {
        return $this->config === '' ? null : $this->config;
    }

    /**
     * @internal
     * @return array<string, string>|null
     */
    public function labels(): ?array
    {
        return $this->tags === [] ? null : $this->tags;
    }

    /**
     * Writes the config of this name, editing it rather than adding a second.
     *
     * @param callable(string, ?RouterConfig): RouterConfigRequest $build
     */
    private function store(callable $build): RouterConfig
    {
        if ($this->config === '') {
            throw new ConfigurationException("configuring writes a named config, so the router needs a name: new Router('healthcare')");
        }
        $held = null;
        foreach (Json::objects(['rows' => $this->client->get('/v1/router/configs')], 'rows') as $row) {
            $config = RouterConfig::fromArray($row);
            if ($config->name === $this->config) {
                $held = $config;
                break;
            }
        }
        $body = $build($this->config, $held)->toArray();
        $stored = $held === null
            ? $this->client->post('/v1/router/configs', body: $body)
            : $this->client->put('/v1/router/configs/{id}', ['id' => $held->id], body: $body);
        return RouterConfig::fromArray(Json::asObject($stored));
    }
}
