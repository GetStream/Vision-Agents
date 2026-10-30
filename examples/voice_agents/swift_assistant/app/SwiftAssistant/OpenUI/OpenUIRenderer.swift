import Foundation
import OpenUI
import StreamChatAI
import SwiftUI

/// Shows the replies an agent writes in OpenUI Lang, with the app's components placeable wherever
/// OpenUI's own content goes.
///
/// ```swift
/// let ui = AIUIHost(components: [PlaceCardComponent()]) {
///     OpenUIRenderer()
/// } send: { action in
///     try await chat.sendMessage(with: action.label, extraData: action.customData)
/// }
/// ```
///
/// Each reply keeps its own runtime, so its forms and the state of its `$variables` survive
/// scrolling. What the person does that is for the agent, such as a follow-up or a form's
/// submit button, reaches the host as an ``AIUIAction`` named after OpenUI's action, with the
/// form's values in its payload. Links open on the device.
@MainActor
final class OpenUIRenderer: AIUIRenderer {
    let format: AIUIFormat = .openUI
    /// The library the agent writes with, before the app's components join it.
    let library: ComponentLibrary
    private let toolProvider: (any OpenUIToolProvider)?

    /// - Parameters:
    ///   - library: The library the agent writes with, the chat library by default.
    ///   - toolProvider: Runs the tools of `Query()` and `Mutation()`. By default they are the
    ///     host's tools, run on the device.
    init(library: ComponentLibrary = .openUIChat, toolProvider: (any OpenUIToolProvider)? = nil) {
        self.library = library
        self.toolProvider = toolProvider
    }

    /// The fence the agent writes its program in, and the instructions its model needs to write
    /// one with the app's components.
    func capabilities(components: [any AIUIComponent]) -> [String: RawJSON] {
        let schemas = components.compactMap { try? Self.schema(of: $0.definition) }
        let library = library.addingContent(schemas.map { ComponentDefinition($0, renderer: nil) })
        return ["fence": .string("openui-lang"), "instructions": .string(library.systemPrompt())]
    }

    func makeEngine(host: any AIUIHosting) -> some AIUIEngine {
        OpenUIEngine(library: library, host: host, toolProvider: toolProvider ?? HostToolProvider(host: host))
    }

    static func schema(of definition: AIUIComponentDefinition) throws -> ComponentSchema {
        try ComponentSchema(portableDefinition: OpenUIJSON.parse(JSONEncoder().encode(definition)))
    }
}

/// Runs the tools of `Query()` and `Mutation()` with the host's tools.
struct HostToolProvider: OpenUIToolProvider {
    let host: any AIUIHosting

    func callTool(_ name: String, arguments: OpenUIObject) async throws -> OpenUIValue {
        try await host.runTool(name, arguments: arguments.rawJSON).openUIValue
    }
}
