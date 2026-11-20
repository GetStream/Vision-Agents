import Foundation
import StreamCore

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
    public let parameters: RawJSON?

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
    public let run: @Sendable ([String: RawJSON]) async throws -> ToolOutput

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

    /// A tool that answers with pictures as well as words, such as a frame from the camera.
    public init(
        name: String,
        description: String,
        parameters: RawJSON? = nil,
        executor: Executor? = nil,
        displayTitle: String? = nil,
        approval: Approval? = nil,
        run: @escaping @Sendable ([String: RawJSON]) async throws -> ToolOutput
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.executor = executor
        self.displayTitle = displayTitle
        self.approval = approval
        self.run = run
    }

    /// A tool that answers in words.
    public init(
        name: String,
        description: String,
        parameters: RawJSON? = nil,
        executor: Executor? = nil,
        displayTitle: String? = nil,
        approval: Approval? = nil,
        run: @escaping @Sendable ([String: RawJSON]) async throws -> String
    ) {
        self.init(
            name: name, description: description, parameters: parameters, executor: executor,
            displayTitle: displayTitle, approval: approval
        ) { ToolOutput(text: try await run($0)) }
    }

    /// A tool whose arguments are decoded into a type of yours, answering with pictures as well
    /// as words. Arguments that do not decode fail the call, and the model is told why.
    public init<Arguments: Decodable>(
        name: String,
        description: String,
        parameters: RawJSON? = nil,
        executor: Executor? = nil,
        displayTitle: String? = nil,
        approval: Approval? = nil,
        run: @escaping @Sendable (Arguments) async throws -> ToolOutput
    ) {
        self.init(
            name: name, description: description, parameters: parameters, executor: executor,
            displayTitle: displayTitle, approval: approval,
            run: { (arguments: [String: RawJSON]) in try await run(Self.decode(arguments)) })
    }

    /// A tool whose arguments are decoded into a type of yours, answering in words. Arguments
    /// that do not decode fail the call, and the model is told why.
    ///
    ///     struct WeatherLookup: Decodable { let location: String }
    ///
    ///     AgentTool(name: "weather_lookup", description: "Look up the weather",
    ///               parameters: .strings(["location": "a city"], required: ["location"])) {
    ///         (args: WeatherLookup) in try await weather.today(args.location)
    ///     }
    public init<Arguments: Decodable>(
        name: String,
        description: String,
        parameters: RawJSON? = nil,
        executor: Executor? = nil,
        displayTitle: String? = nil,
        approval: Approval? = nil,
        run: @escaping @Sendable (Arguments) async throws -> String
    ) {
        self.init(
            name: name, description: description, parameters: parameters, executor: executor,
            displayTitle: displayTitle, approval: approval,
            run: { (arguments: Arguments) in ToolOutput(text: try await run(arguments)) })
    }

    private static func decode<Arguments: Decodable>(_ arguments: [String: RawJSON]) throws -> Arguments {
        do {
            return try JSONDecoder().decode(Arguments.self, from: JSONEncoder().encode(arguments))
        } catch let error as DecodingError {
            throw ToolArgumentsError(failure: error)
        }
    }
}

/// Arguments the model sent that do not decode into a tool's type, said in words the model can
/// correct them from: `DecodingError`'s own description names no field.
struct ToolArgumentsError: LocalizedError {
    let failure: DecodingError

    var errorDescription: String? {
        switch failure {
        case .keyNotFound(let key, let context):
            return "the arguments have no \"\(Self.path(context.codingPath + [key]))\""
        case .typeMismatch(_, let context), .valueNotFound(_, let context), .dataCorrupted(let context):
            let path = Self.path(context.codingPath)
            return path.isEmpty
                ? "the arguments do not fit the tool: \(context.debugDescription)"
                : "the arguments do not fit the tool at \"\(path)\": \(context.debugDescription)"
        @unknown default:
            return failure.localizedDescription
        }
    }

    private static func path(_ keys: [any CodingKey]) -> String {
        keys.map { $0.intValue.map(String.init) ?? $0.stringValue }.joined(separator: ".")
    }
}

extension AgentTool {
    var schema: SessionTool {
        SessionTool(
            approval: approval.map {
                .init(
                    allowTitle: $0.allowTitle, declineTitle: $0.declineTitle,
                    message: $0.message, reasonArgument: $0.reasonArgument, title: $0.title)
            },
            description: description,
            displayTitle: displayTitle,
            executor: executor.map { .init(rawValue: $0.rawValue) },
            name: name,
            // A schema that is not an object describes no arguments the router can read.
            parameters: parameters.map { $0.dictionaryValue ?? [:] })
    }
}

/// What a tool gives the model: words, and pictures beside them, such as a photo the person
/// picked or a frame from the camera.
///
/// The router hands a result's pictures to the agent's `vision` skill on a call, which looks at
/// them and tells the conversation what it found. With no such worker, as in a chat, a model
/// that sees images is shown them with its next reply alone; one that reads only text is told
/// the tool failed. Each picture is a data URI or an https URL, and one socket message holds at
/// most 5 MB, so shrink a photo before returning it.
public struct ToolOutput: Sendable, Hashable {
    /// What the tool says.
    public var text: String
    /// What the tool shows.
    public var images: [ImageSource]

    public init(text: String = "", images: [ImageSource] = []) {
        self.text = text
        self.images = images
    }
}

extension RawJSON {
    /// A JSON Schema object for a tool whose arguments are all strings.
    ///
    /// A convenience for the common shape, so declaring a tool does not mean writing out a
    /// schema by hand. Anything else, write the object yourself.
    ///
    ///     .strings(["location": "the city, e.g. Boulder, CO"], required: ["location"])
    public static func strings(
        _ properties: [String: String],
        required: [String] = []
    ) -> RawJSON {
        .dictionary([
            "type": .string("object"),
            "properties": .dictionary(
                properties.mapValues { description in
                    .dictionary(["type": .string("string"), "description": .string(description)])
                }),
            "required": .array(required.map { .string($0) }),
        ])
    }
}
