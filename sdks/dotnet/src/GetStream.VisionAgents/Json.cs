using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;

namespace GetStream.VisionAgents;

/// <summary>
/// The one set of serializer options every request, response and frame goes through.
/// </summary>
public static class Json
{
    /// <summary>
    /// Web defaults, with nulls left out of what is written.
    /// </summary>
    /// <remarks>
    /// Leaving a null out is what "not set" means on the wire: the config or the router
    /// decides. Writing it would be sending a value somebody did not choose. The resolver is
    /// named because a tool's schema is written from these options (<see cref="Tools"/>), and
    /// the schema exporter refuses options no serialization has given one yet, which a tool
    /// registered before the first request would otherwise hit.
    /// </remarks>
    public static JsonSerializerOptions Options { get; } = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver(),
    };
}
