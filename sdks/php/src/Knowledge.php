<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\KnowledgeUrl;
use GetStream\VisionAgents\Generated\KnowledgeUrlRequest;
use GetStream\VisionAgents\Generated\KnowledgeUrlState;

/**
 * An agent's knowledge base, as somewhere to put more of it.
 *
 * The namespace is the agent's own name, which is where the knowledge in a synced directory
 * lands, so what is added here is found by the same lookup mid-answer.
 */
final readonly class Knowledge
{
    /**
     * @param float $timeout seconds a page is waited for before it is returned still pending
     */
    public function __construct(
        private Client $client,
        public string $namespace,
        private float $timeout = 180.0,
        private float $pollEvery = 0.25,
    ) {
    }

    /**
     * Keeps the knowledge base filled from a page published elsewhere.
     *
     * The router queues the read and cuts the page into passages; this waits for it, so what
     * comes back already says whether it worked. It stays a subscription: the passages are
     * keyed by the url, and reading it again replaces them.
     */
    public function addUrl(string $url, string $title = '', string $description = ''): KnowledgeUrl
    {
        if ($this->namespace === '') {
            throw new ConfigurationException('a knowledge base is named by the agent it belongs to, and this one has no name');
        }
        $body = new KnowledgeUrlRequest($this->namespace, $url, $title === '' ? null : $title, $description === '' ? null : $description);
        $page = KnowledgeUrl::fromArray(Json::asObject($this->client->post('/v1/agents/knowledge/urls', body: $body->toArray())));

        $deadline = microtime(true) + $this->timeout;
        while ($page->state === KnowledgeUrlState::Pending && microtime(true) < $deadline) {
            Pause::for($this->pollEvery);
            $page = KnowledgeUrl::fromArray(Json::asObject($this->client->get('/v1/agents/knowledge/urls/{id}', ['id' => $page->id])));
        }
        return $page;
    }
}
