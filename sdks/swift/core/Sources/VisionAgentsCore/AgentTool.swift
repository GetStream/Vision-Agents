import Foundation

/// A function of yours the agent can call.
///
/// The agent runs in the backend but the tool runs here, which is the point: a tool can read
/// the signed-in user's data, or something only the phone knows, without any of it leaving
/// the device. The router asks over the session socket and waits for the answer.
public struct AgentTool: Sendable {
    /// What the model calls it. Must be unique within a session.
    public let name: String

    /// What it is for, in words the model reads to decide whether to call it.
    public let description: String

    /// A JSON Schema object describing the arguments, or nil for a tool that takes none.
    public let parameters: JSONValue?

    /// Who runs it, as the people in a persistent conversation are shown. Nil is the server.
    public let executor: Executor?

    /// What a call is doing, in words for the people in the conversation, such as "Checking
    /// your location". At most 80 characters.
    public let displayTitle: String?

    /// What a person is asked before each call runs, or nil to run every call straight away.
    /// A call that asks waits in `AgentSession.approvals` until `decide` answers it.
    public let approval: Approval?

    /// Runs the tool. What it returns is given to the model as the result; throwing tells the
    /// model the tool failed and why.
    public let run: @Sendable ([String: JSONValue]) async throws -> String

    public enum Executor: String, Sendable {
        case server
        /// A person's device: the call is shown as awaiting it until it answers.
        case client
    }

    /// A tool's question to the person whose message a call answers.
    ///
    /// In a conversation kept in Stream Chat the call's step carries it as `awaiting_approval`,
    /// which Stream's AI components ask from, and every channel member can read it: keep
    /// anything private out of it. A client tool is asked about only once the router knows the
    /// install to address it to; until then, declare it with the server executor.
    public struct Approval: Sendable, Hashable {
        /// The question, such as "Share your location?". At most 80 characters.
        public var title: String
        /// What allowing it shares or does, such as "Only your city is shared."
        public var message: String?
        /// The string argument in which the model says why it wants the call, shown beside the
        /// question. Declare it in `parameters` so the model fills it in.
        public var reasonArgument: String?
        /// The label of the button that allows the call.
        public var allowTitle: String?
        /// The label of the button that declines it.
        public var declineTitle: String?

        public init(
            title: String,
            message: String? = nil,
            reasonArgument: String? = nil,
            allowTitle: String? = nil,
            declineTitle: String? = nil
        ) {
            self.title = title
            self.message = message
            self.reasonArgument = reasonArgument
            self.allowTitle = allowTitle
            self.declineTitle = declineTitle
        }
    }

    public init(
        name: String,
        description: String,
        parameters: JSONValue? = nil,
        executor: Executor? = nil,
        displayTitle: String? = nil,
        approval: Approval? = nil,
        run: @escaping @Sendable ([String: JSONValue]) async throws -> String
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.executor = executor
        self.displayTitle = displayTitle
        self.approval = approval
        self.run = run
    }
}

extension JSONValue {
    /// A JSON Schema object for a tool whose arguments are all strings.
    ///
    /// A convenience for the common shape, so declaring a tool does not mean writing out a
    /// schema by hand. Anything else, write the object yourself.
    ///
    ///     .strings(["location": "the city, e.g. Boulder, CO"], required: ["location"])
    public static func strings(
        _ properties: [String: String],
        required: [String] = []
    ) -> JSONValue {
        .object([
            "type": .string("object"),
            "properties": .object(
                properties.mapValues { description in
                    .object(["type": .string("string"), "description": .string(description)])
                }),
            "required": .array(required.map { .string($0) }),
        ])
    }
}
