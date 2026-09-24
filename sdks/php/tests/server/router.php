<?php

declare(strict_types=1);

// A router for `php -S`: it answers from a script the test wrote and writes down every request,
// so a test asserts on what was sent rather than on which method was called.
//
// State lives in the directory named by LOCAL_ROUTER_STATE:
//   script.json    {"METHOD /path": [{"status": 200, "body": ..., "headers": {...}}, ...]}
//                  answered in order, the last one repeated once the rest are used up
//   served.json    how many of each have been used
//   requests.jsonl one line per request: method, path, query, headers, body

$state = getenv('LOCAL_ROUTER_STATE');
if (!is_string($state) || $state === '') {
    http_response_code(500);
    echo '{"error":"LOCAL_ROUTER_STATE is not set"}';
    return true;
}

$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$uri = $_SERVER['REQUEST_URI'] ?? '/';
$path = parse_url($uri, PHP_URL_PATH);
$path = is_string($path) ? $path : '/';
$query = parse_url($uri, PHP_URL_QUERY);
$body = file_get_contents('php://input');

$lock = fopen("{$state}/lock", 'c');
if ($lock === false) {
    http_response_code(500);
    return true;
}
flock($lock, LOCK_EX);

file_put_contents("{$state}/requests.jsonl", json_encode([
    'method' => $method,
    'path' => $path,
    'query' => is_string($query) ? $query : '',
    'headers' => array_change_key_case(getallheaders()),
    'body' => $body === false ? '' : $body,
], JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES) . "\n", FILE_APPEND);

$script = is_file("{$state}/script.json") ? json_decode((string) file_get_contents("{$state}/script.json"), true) : [];
$served = is_file("{$state}/served.json") ? json_decode((string) file_get_contents("{$state}/served.json"), true) : [];
$key = "{$method} {$path}";
$answers = is_array($script) && isset($script[$key]) && is_array($script[$key]) ? $script[$key] : [];
$count = is_array($served) && isset($served[$key]) && is_int($served[$key]) ? $served[$key] : 0;
$served = is_array($served) ? $served : [];
$served[$key] = $count + 1;
file_put_contents("{$state}/served.json", json_encode($served, JSON_THROW_ON_ERROR));

flock($lock, LOCK_UN);
fclose($lock);

if ($answers === []) {
    http_response_code(404);
    header('Content-Type: application/json');
    echo json_encode(['error' => "nothing scripted for {$key}"]);
    return true;
}

$answer = $answers[min($count, count($answers) - 1)];
http_response_code((int) ($answer['status'] ?? 200));
foreach (($answer['headers'] ?? []) as $name => $value) {
    header("{$name}: {$value}");
}
if (array_key_exists('body', $answer) && $answer['body'] !== null) {
    header('Content-Type: application/json');
    echo is_string($answer['body']) && ($answer['raw'] ?? false) ? $answer['body'] : json_encode($answer['body'], JSON_UNESCAPED_SLASHES);
}
return true;
