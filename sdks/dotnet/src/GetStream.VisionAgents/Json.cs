using System.Text.Json;
using System.Text.Json.Serialization;

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
    /// decides. Writing it would be sending a value somebody did not choose.
    /// </remarks>
    public static JsonSerializerOptions Options { get; } = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}
