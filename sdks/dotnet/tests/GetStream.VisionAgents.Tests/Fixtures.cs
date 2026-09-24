using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace GetStream.VisionAgents.Tests;

/// <summary>What the router answers with, in the shapes its spec gives them.</summary>
internal static class Fixtures
{
    public const string Secret = "a-stream-secret-is-sixty-four-characters-long-so-this-one-is-too";

    public static object Session(string id = "s1", string callId = "", bool persisted = false) => new
    {
        id,
        call_id = callId,
        call_type = "agent",
        user_id = "jean",
        agent_id = "jean",
        state = "running",
        created_at = "2026-09-24T10:00:00Z",
        conversation_id = persisted ? "agent:support-1" : null,
    };

    public static object Config(string name = "jean", string id = "cfg-1") => new
    {
        id,
        name,
        mode = "voice",
        created_at = "2026-09-24T10:00:00Z",
        updated_at = "2026-09-24T10:00:00Z",
    };

    public static object StreamCall(string id) => new
    {
        call = new { id, type = "agent", cid = $"agent:{id}", created_at = "2026-09-24T10:00:00Z", updated_at = "2026-09-24T10:00:00Z" },
        created = true,
        duration = "1ms",
        members = Array.Empty<object>(),
        own_capabilities = Array.Empty<object>(),
    };

    public static VisionAgentsClient Client(TestRouter router) =>
        new(new VisionAgentsOptions { Url = router.Url, CustomerId = "examples" });

    public static Edge Edge(TestRouter router) =>
        new(new EdgeOptions { ApiKey = "key", ApiSecret = Secret, BaseUrl = router.Url, MonitorUrl = "https://demo.test" });

    /// <summary>A token's claims, once its signature is checked against the secret.</summary>
    public static JsonObject Claims(string bearer, string secret = Secret)
    {
        var parts = bearer["Bearer ".Length..].Split('.');
        var signature = HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), Encoding.UTF8.GetBytes($"{parts[0]}.{parts[1]}"));
        Assert.Equal(Base64Url(signature), parts[2]);
        return JsonNode.Parse(FromBase64Url(parts[1]))!.AsObject();
    }

    public static string Text(this JsonNode? node, string name) => node?[name]?.GetValue<string>() ?? "";

    public static JsonElement Element(object value) => JsonSerializer.SerializeToElement(value);

    private static string Base64Url(byte[] bytes) => Convert.ToBase64String(bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_');

    private static byte[] FromBase64Url(string text)
    {
        var padded = text.Replace('-', '+').Replace('_', '/');
        return Convert.FromBase64String(padded + new string('=', (4 - padded.Length % 4) % 4));
    }
}
