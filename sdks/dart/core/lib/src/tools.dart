/// Runs a tool. What it returns is given to the model as the result; throwing tells the model
/// the tool failed, and why.
typedef ToolHandler = Future<String> Function(Map<String, Object?> arguments);

/// A function of yours the agent can call.
///
/// The agent runs in the backend but the tool runs here, which is the point: a tool can read
/// the signed-in user's data, or something only the device knows, without any of it leaving
/// the device. The router asks over the session socket and waits for the answer.
final class AgentTool {
  const AgentTool({
    required this.name,
    required this.description,
    required this.run,
    this.parameters,
  });

  /// What the model calls it. Must be unique within a session.
  final String name;

  /// What it is for, in the words the model reads to decide whether to call it.
  final String description;

  /// A JSON Schema object describing the arguments, or null for a tool that takes none.
  final Map<String, Object?>? parameters;
  final ToolHandler run;

  /// A JSON Schema object for a tool whose arguments are all strings.
  ///
  /// A convenience for the common shape; for anything else, write the object yourself.
  ///
  ///     AgentTool.strings({'location': 'the city, e.g. Boulder, CO'}, required: ['location'])
  static Map<String, Object?> strings(
    Map<String, String> properties, {
    List<String> required = const [],
  }) => {
    'type': 'object',
    'properties': {
      for (final MapEntry(key: name, value: description) in properties.entries)
        name: {'type': 'string', 'description': description},
    },
    'required': required,
  };
}
