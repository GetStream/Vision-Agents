import Foundation

/// One JSON value, as it arrived.
///
/// Session frames are kept in this form rather than decoded into a fixed struct per event so
/// that a frame this version of the SDK has never heard of still reaches the caller whole. It
/// is the same bargain the Go SDK makes with its `Frame map[string]any`.
public enum JSONValue: Sendable, Hashable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}

extension JSONValue: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "not a JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

extension JSONValue {
    /// The string this holds, or the empty string for anything else.
    ///
    /// Absent and empty are one case on purpose: the router writes `""` for a field that does
    /// not apply, so a caller that told them apart would be distinguishing nothing.
    public var stringValue: String {
        if case .string(let value) = self { return value }
        return ""
    }

    /// The number this holds, rounded, or nil for anything else.
    public var intValue: Int? {
        if case .number(let value) = self { return Int(value) }
        return nil
    }

    /// The number this holds, or nil for anything else.
    public var doubleValue: Double? {
        if case .number(let value) = self { return value }
        return nil
    }

    /// Whether this is `true`. Anything else, including absence, is false.
    public var boolValue: Bool {
        if case .bool(let value) = self { return value }
        return false
    }

    /// The members of this object, or an empty dictionary for anything else.
    public var objectValue: [String: JSONValue] {
        if case .object(let value) = self { return value }
        return [:]
    }

    /// The members of this array, or an empty array for anything else.
    public var arrayValue: [JSONValue] {
        if case .array(let value) = self { return value }
        return []
    }
}
