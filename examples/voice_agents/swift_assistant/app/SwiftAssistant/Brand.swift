import StreamChatA2UI
import StreamChatAI
import StreamChatOpenUI
import SwiftUI

/// The brand the agent's UI wears: its accent, and how round its corners are.
///
/// Settings chooses one, and the conversation shows it at once: its UI's renderers are made
/// again with the brand's theme, for OpenUI and A2UI alike, and the app's own components read it
/// from the environment. The agent writes the same UI whichever is chosen.
enum Brand: String, CaseIterable, Identifiable {
    case stream
    case ember
    case nordic

    /// Where Settings keeps the choice.
    static let storageKey = "brand"

    /// The brand Settings chose.
    static var chosen: Brand {
        UserDefaults.standard.string(forKey: storageKey).flatMap(Brand.init) ?? .stream
    }

    var id: Self { self }

    var title: String {
        switch self {
        case .stream: "Stream Default"
        case .ember: "Ember Hospitality"
        case .nordic: "Nordic Mobility"
        }
    }

    var subtitle: String {
        switch self {
        case .stream: "Electric blue accent · 16pt rounded cards"
        case .ember: "Warm terracotta accent · 10pt compact cards"
        case .nordic: "Crisp teal accent · 6pt architectural edges"
        }
    }

    var accent: Color {
        switch self {
        case .stream: Color(red: 0x00 / 255, green: 0x5F / 255, blue: 0xFF / 255)
        case .ember: Color(red: 0xD9 / 255, green: 0x5D / 255, blue: 0x24 / 255)
        case .nordic: Color(red: 0x0F / 255, green: 0x76 / 255, blue: 0x6E / 255)
        }
    }

    /// The corners of a card. Smaller parts round less, in proportion.
    var cornerRadius: CGFloat {
        switch self {
        case .stream: 16
        case .ember: 10
        case .nordic: 6
        }
    }

    /// The conversation's OpenUI theme in this brand.
    @MainActor
    var openUITheme: OpenUITheme {
        var theme = OpenUITheme(appearance: .shared)
        theme.colors.accent = accent
        theme.colors.info = accent
        theme.radius.small = cornerRadius / 2
        theme.radius.medium = cornerRadius * 0.75
        theme.radius.large = cornerRadius
        return theme
    }

    /// The conversation's A2UI theme in this brand.
    @MainActor
    var a2uiTheme: A2UITheme {
        var theme = A2UITheme(appearance: .shared)
        theme.accent = accent
        theme.cornerRadius = cornerRadius
        theme.buttonCornerRadius = cornerRadius * 0.75
        theme.controlCornerRadius = cornerRadius / 2
        return theme
    }
}

extension EnvironmentValues {
    /// The brand the app's components in the agent's UI wear.
    @Entry var brand: Brand = .stream
}
