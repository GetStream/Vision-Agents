<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\AgentResponse as ResponseRow;
use GetStream\VisionAgents\Generated\AgentResponseItem;
use GetStream\VisionAgents\Generated\AgentResponsePage;
use GetStream\VisionAgents\Generated\CreateResponseRequest;
use GetStream\VisionAgents\Generated\ImageSource;
use GetStream\VisionAgents\Generated\RewindSessionRequest;

/**
 * A session's turns.
 *
 * `items` here is the whole conversation flattened, which is how a conversation reads: the
 * question, what the agent did about it, what it said, then the next question. One turn's own
 * items come off the handle `create` returns.
 */
final readonly class Responses
{
    public Items $items;

    public function __construct(private Client $client, private string $sessionId)
    {
        $this->items = new Items($client, $sessionId);
    }

    /**
     * Asks the agent something and names the turn it answers as.
     *
     * It returns as soon as the agent has started answering, not when it has finished: a model
     * takes seconds, and a request that waited them out would time out on anything worth asking.
     *
     * A question that is text only is sent with a fresh request id; one with images is sent
     * with none, because the router refuses one with media.
     *
     * @param list<ImageSource> $images
     */
    public function create(string $text, array $images = []): AgentResponse
    {
        return $this->createAnswering($text, $images, '');
    }

    /**
     * @internal for Dispatch, which answers an inbound message on the request it was sent as
     *
     * @param list<ImageSource> $images
     * @param string $requestId the request id of the inbound message being answered; empty generates one
     */
    public function createAnswering(string $text, array $images, string $requestId): AgentResponse
    {
        $requestId = $images !== [] ? null : ($requestId === '' ? bin2hex(random_bytes(16)) : $requestId);
        $body = new CreateResponseRequest(text: $text, images: $images === [] ? null : $images, requestId: $requestId);
        $created = $this->client->post('/v1/agents/sessions/{id}/responses', ['id' => $this->sessionId], body: $body->toArray());
        return new AgentResponse($this->client, ResponseRow::fromArray(Json::asObject($created)));
    }

    /**
     * Goes back to a response and carries on from there, as though nothing after it was said.
     *
     * An item stands for the response it belongs to, so what a transcript renders is enough to
     * rewind to. A conversation kept in Stream Chat is refused, because the channel would still
     * hold the later turns: fork it at the response instead.
     */
    public function rewind(AgentResponse|ResponseRow|AgentResponseItem|string $to): void
    {
        $responseId = match (true) {
            is_string($to) => $to,
            $to instanceof AgentResponseItem => $to->responseId,
            $to instanceof AgentResponse => $to->id(),
            default => $to->id,
        };
        if ($responseId === '') {
            throw new ConfigurationException('a response that was never recorded cannot be rewound to');
        }
        $this->client->post('/v1/agents/sessions/{id}/rewind', ['id' => $this->sessionId], body: (new RewindSessionRequest($responseId))->toArray());
    }

    /**
     * A page of the turns so far, oldest first. Pass its `nextCursor` back for the next one.
     */
    public function list(?int $limit = null, ?string $cursor = null): AgentResponsePage
    {
        $listed = $this->client->get('/v1/agents/sessions/{id}/responses', ['id' => $this->sessionId], ['limit' => $limit, 'cursor' => $cursor]);
        return AgentResponsePage::fromArray(Json::asObject($listed));
    }
}
