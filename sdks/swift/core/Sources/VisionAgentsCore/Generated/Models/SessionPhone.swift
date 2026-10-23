import Foundation
import StreamCore

/// The number the session acts from, which is what turns transferring on.
final class SessionPhone: Sendable, Codable, JSONEncodable {
    /// One of the customer's own numbers, written as +15551234567.
    let number: String
    /// Who carries an outbound leg.
    let vendor: String?
    /// The outbound leg, set for a call the agent placed. Without one the agent has no keypad to press at.
    let vendorCallId: String?

    init(number: String, vendor: String? = nil, vendorCallId: String? = nil) {
        self.number = number
        self.vendor = vendor
        self.vendorCallId = vendorCallId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.number = try container.decode(String.self, forKey: .number)
        self.vendor = try container.decodeIfPresent(String.self, forKey: .vendor)
        self.vendorCallId = try container.decodeIfPresent(String.self, forKey: .vendorCallId)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(number, forKey: .number)
        try container.encodeIfPresent(vendor, forKey: .vendor)
        try container.encodeIfPresent(vendorCallId, forKey: .vendorCallId)
    }

}

extension SessionPhone: Hashable {
    static func == (lhs: SessionPhone, rhs: SessionPhone) -> Bool {
        lhs.number == rhs.number &&
        lhs.vendor == rhs.vendor &&
        lhs.vendorCallId == rhs.vendorCallId
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(number)
        hasher.combine(vendor)
        hasher.combine(vendorCallId)
    }
}
