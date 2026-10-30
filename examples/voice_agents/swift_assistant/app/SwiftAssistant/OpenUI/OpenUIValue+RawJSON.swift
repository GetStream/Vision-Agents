import Foundation
import OpenUI
import StreamChatAI

extension OpenUIValue {
    /// The value as JSON. Components and expressions turn into their plain JSON shapes.
    var rawJSON: RawJSON {
        switch self {
        case .null:
            return .nil
        case let .bool(value):
            return .bool(value)
        case let .number(value):
            return .number(value)
        case let .string(value):
            return .string(value)
        case let .array(items):
            return .array(items.map(\.rawJSON))
        case let .object(object):
            return .dictionary(object.rawJSON)
        case .element, .expression:
            let plain = plainJSON
            if case .element = plain { return .nil }
            if case .expression = plain { return .nil }
            return plain.rawJSON
        }
    }
}

extension OpenUIObject {
    var rawJSON: [String: RawJSON] {
        Dictionary(entries.map { ($0.key, $0.value.rawJSON) }, uniquingKeysWith: { _, last in last })
    }

    init(rawJSON: [String: RawJSON]) {
        self.init(rawJSON.map { ($0.key, $0.value.openUIValue) })
    }
}

extension RawJSON {
    var openUIValue: OpenUIValue {
        switch self {
        case .nil:
            return .null
        case let .bool(value):
            return .bool(value)
        case let .number(value):
            return .number(value)
        case let .string(value):
            return .string(value)
        case let .array(items):
            return .array(items.map(\.openUIValue))
        case let .dictionary(fields):
            return .object(OpenUIObject(rawJSON: fields))
        }
    }
}
