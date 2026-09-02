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

    /// Runs the tool. What it returns is given to the model as the result; throwing tells the
    /// model the tool failed and why.
    public let run: @Sendable ([String: JSONValue]) async throws -> String

    public init(
        name: String,
        description: String,
        parameters: JSONValue? = nil,
        run: @escaping @Sendable ([String: JSONValue]) async throws -> String
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
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
