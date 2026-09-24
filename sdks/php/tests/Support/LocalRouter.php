<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Support;

use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Json;
use RuntimeException;

/**
 * A real HTTP server, `php -S` running tests/server/router.php, answering from a script.
 */
final class LocalRouter
{
    public readonly string $url;
    private readonly string $state;
    /** @var resource */
    private $process;

    public function __construct()
    {
        $this->state = sys_get_temp_dir() . '/local-router-' . bin2hex(random_bytes(6));
        mkdir($this->state);
        $port = self::freePort();
        $this->url = "http://127.0.0.1:{$port}";
        $process = proc_open(
            [PHP_BINARY, '-S', "127.0.0.1:{$port}", __DIR__ . '/../server/router.php'],
            [0 => ['pipe', 'r'], 1 => ['file', '/dev/null', 'w'], 2 => ['file', '/dev/null', 'w']],
            $pipes,
            null,
            ['LOCAL_ROUTER_STATE' => $this->state, 'PHP_CLI_SERVER_WORKERS' => '4'],
        );
        if ($process === false) {
            throw new RuntimeException('php -S did not start');
        }
        $this->process = $process;
        $deadline = microtime(true) + 5;
        while (microtime(true) < $deadline) {
            $socket = @fsockopen('127.0.0.1', $port, $code, $message, 0.1);
            if ($socket !== false) {
                fclose($socket);
                return;
            }
            usleep(20_000);
        }
        throw new RuntimeException("php -S never listened on {$port}");
    }

    /**
     * A client for this server, as a customer of a router that trusts the header.
     */
    public function client(string $customerId = 'examples'): Client
    {
        return new Client(new Backend(url: $this->url, customerId: $customerId));
    }

    /**
     * Scripts what `METHOD /path` answers, in order; the last answer repeats.
     *
     * @param mixed ...$bodies each one a JSON body answered with `$status`
     */
    public function answer(string $method, string $path, int $status, mixed ...$bodies): self
    {
        $script = $this->script();
        $key = "{$method} {$path}";
        foreach ($bodies === [] ? [null] : $bodies as $body) {
            $script[$key][] = ['status' => $status, 'body' => $body];
        }
        file_put_contents("{$this->state}/script.json", Json::encode($script));
        return $this;
    }

    /**
     * @param array<string, string> $headers
     */
    public function answerWithHeaders(string $method, string $path, int $status, mixed $body, array $headers): self
    {
        $script = $this->script();
        $script["{$method} {$path}"][] = ['status' => $status, 'body' => $body, 'headers' => $headers];
        file_put_contents("{$this->state}/script.json", Json::encode($script));
        return $this;
    }

    /**
     * Everything received so far, oldest first.
     *
     * @return list<Received>
     */
    public function received(): array
    {
        $file = "{$this->state}/requests.jsonl";
        if (!is_file($file)) {
            return [];
        }
        $lines = file($file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        return array_map(static fn (string $line): Received => Received::fromLine($line), $lines === false ? [] : $lines);
    }

    /**
     * The requests to one path.
     *
     * @return list<Received>
     */
    public function to(string $method, string $path): array
    {
        return array_values(array_filter($this->received(), static fn (Received $r): bool => $r->method === $method && $r->path === $path));
    }

    public function stop(): void
    {
        proc_terminate($this->process);
        proc_close($this->process);
        exec('rm -rf ' . escapeshellarg($this->state));
    }

    /**
     * @return array<string, list<array<string, mixed>>>
     */
    private function script(): array
    {
        $file = "{$this->state}/script.json";
        if (!is_file($file)) {
            return [];
        }
        /** @var array<string, list<array<string, mixed>>> */
        return Json::decode((string) file_get_contents($file));
    }

    private static function freePort(): int
    {
        $server = stream_socket_server('tcp://127.0.0.1:0');
        if ($server === false) {
            throw new RuntimeException('no free port');
        }
        $name = (string) stream_socket_get_name($server, false);
        fclose($server);
        return (int) substr($name, (int) strrpos($name, ':') + 1);
    }
}
