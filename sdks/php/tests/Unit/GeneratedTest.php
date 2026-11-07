<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Generated\ImageSource;
use GetStream\VisionAgents\Generated\RespondRequest;
use PHPUnit\Framework\TestCase;

final class GeneratedTest extends TestCase
{
    public function testAPropertyAddedToTheSpecKeepsTheReleasedPositionalArguments(): void
    {
        // images came after request_id; alphabetically it would have taken request_id's place.
        $request = new RespondRequest('question', null, 'device-1', 'request-1');
        $withPhoto = new RespondRequest('question', null, 'device-1', 'request-1', [new ImageSource(url: 'data:image/png;base64,AA==')]);

        self::assertSame(['text' => 'question', 'client_id' => 'device-1', 'request_id' => 'request-1'], $request->toArray());
        self::assertSame('request-1', $withPhoto->requestId);
        self::assertSame([['url' => 'data:image/png;base64,AA==']], $withPhoto->toArray()['images']);
    }
}
