import Foundation
import StreamChatAI

extension RawJSON {
    /// A value of A2UI's data model or of a resolved property, as JSON.
    init(a2ui value: Any?) {
        switch value {
        case let string as String:
            self = .string(string)
        case let number as NSNumber:
            // JSON's booleans come as numbers too, which only their type tells apart.
            self = CFGetTypeID(number) == CFBooleanGetTypeID() ? .bool(number.boolValue) : .number(number.doubleValue)
        case let array as [Any]:
            self = .array(array.map(RawJSON.init(a2ui:)))
        case let object as [String: Any]:
            self = .dictionary(object.mapValues(RawJSON.init(a2ui:)))
        default:
            self = .nil
        }
    }
}
