/// Runs a tool. What it returns is given to the model as the result; throwing tells the model
/// the tool failed, and why.
typedef ToolHandler = Future<String> Function(Map<String, Object?> arguments);

/// Who runs a tool, as a persistent conversation shows it.
enum ToolExecutor {
  /// The caller's backend. What the router assumes when nothing is said.
  server,

  /// A person's device: the call is shown as awaiting the device of the person whose command
  /// it answers, with its arguments, which every channel member can read.
  client,
}

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
    this.displayTitle,
    this.executor,
    this.approval,
  });

  /// What the model calls it. Must be unique within a session.
  final String name;

  /// What it is for, in the words the model reads to decide whether to call it.
  final String description;

  /// A JSON Schema object describing the arguments, or null for a tool that takes none.
  final Map<String, Object?>? parameters;
  final ToolHandler run;

  /// What a call is doing, in words for the people in the conversation, such as "Checking
  /// your location". Up to 80 characters, shown on the reply's tool call attachment.
  final String? displayTitle;

  /// Who runs it. Null leaves it to the router, which is [ToolExecutor.server].
  final ToolExecutor? executor;

  /// What a person is asked before each call runs, or null to run every call straight away.
  /// A call that asks waits in `AgentSession.approvals` until `AgentSession.decide` answers it.
  final ToolApproval? approval;

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

/// A tool's question to the person whose message a call answers.
///
/// In a conversation kept in Stream Chat the call's step carries it as `awaiting_approval`,
/// and every channel member can read it: keep anything private out of it. A client tool is
/// asked about only once the router knows the install to address it to; until then, declare
/// it with [ToolExecutor.server].
final class ToolApproval {
  const ToolApproval({
    required this.title,
    this.message,
    this.reasonArgument,
    this.allowTitle,
    this.declineTitle,
  });

  /// The question, such as "Share your location?". At most 80 characters.
  final String title;

  /// What allowing it shares or does, such as "Only your city is shared."
  final String? message;

  /// The string argument in which the model says why it wants the call, shown beside the
  /// question. Declare it in `parameters` so the model fills it in.
  final String? reasonArgument;

  /// The label of the button that allows the call.
  final String? allowTitle;

  /// The label of the button that declines it.
  final String? declineTitle;
}
