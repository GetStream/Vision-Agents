import Foundation
import StreamCore

/// Tools to give a running session's caller.
final class AddSessionToolsRequest: Sendable, Codable, JSONEncodable {
    /// The tools, each declared as it is when a session is created. One named as a tool the caller already has replaces it.
    let tools: [SessionTool]?

    init(tools: [SessionTool]? = nil) {
        self.tools = tools
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.tools = try container.decodeIfPresent([SessionTool].self, forKey: .tools)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(tools, forKey: .tools)
    }

}

extension AddSessionToolsRequest: Hashable {
    static func == (lhs: AddSessionToolsRequest, rhs: AddSessionToolsRequest) -> Bool {
        lhs.tools == rhs.tools
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(tools)
    }
}
