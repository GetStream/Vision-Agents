import Foundation
import OpenUI
import StreamChatAI
import SwiftUI

/// The OpenUI replies of one conversation, each with a runtime of its own.
@MainActor
final class OpenUIEngine: AIUIEngine {
    private weak var host: (any AIUIHosting)?
    private let toolProvider: any OpenUIToolProvider
    private var library: ComponentLibrary
    private(set) var runtimes: [String: OpenUIRuntime] = [:]
    private var messageIDs: [ObjectIdentifier: String] = [:]

    init(library: ComponentLibrary, host: any AIUIHosting, toolProvider: any OpenUIToolProvider) {
        self.host = host
        self.toolProvider = toolProvider
        self.library = library
        let components = host.components.compactMap { component -> ComponentDefinition? in
            guard let schema = try? OpenUIRenderer.schema(of: component.definition) else { return nil }
            return definition(of: component, schema: schema)
        }
        self.library = library.withEntrances.addingContent(components)
    }

    func view(for document: AIUIDocument, messageID: String) -> OpenUIMessageView? {
        guard let program = document.text else { return nil }
        let runtime = runtimes[messageID] ?? makeRuntime(messageID: messageID, program: program, isStreaming: document.isStreaming)
        return OpenUIMessageView(runtime: runtime, program: program, isStreaming: document.isStreaming)
    }

    // A reply's runtime starts with its program, before any view follows it.
    private func makeRuntime(messageID: String, program: String, isStreaming: Bool) -> OpenUIRuntime {
        let state = host?.state(for: messageID)?.dictionaryValue.map(OpenUIObject.init(rawJSON:))
        let runtime = OpenUIRuntime(library: library, toolProvider: toolProvider, initialState: state)
        runtime.onAction = { [weak self] event in
            self?.handle(event, messageID: messageID)
        }
        runtime.onStateUpdate = { [weak self] state in
            self?.host?.setState(.dictionary(state.rawJSON), for: messageID)
        }
        runtimes[messageID] = runtime
        messageIDs[ObjectIdentifier(runtime)] = messageID
        runtime.update(response: program, isStreaming: isStreaming)
        return runtime
    }

    // The OpenUI component that shows one of the app's, with the app's view as it is.
    private func definition(of component: some AIUIComponent, schema: ComponentSchema) -> ComponentDefinition {
        ComponentDefinition(schema) { [weak self] context in
            self?.view(of: component, in: context)
        }
    }

    private func view(of component: some AIUIComponent, in context: ComponentContext) -> (some View)? {
        self.context(of: component, in: context).flatMap { componentContext in
            component.view(arguments: context.props.values.rawJSON, context: componentContext)
        }
    }

    // What one of the app's components gets besides its arguments, while the host is there.
    private func context(of component: some AIUIComponent, in context: ComponentContext) -> AIUIComponentContext? {
        guard let host else { return nil }
        let messageID = messageIDs[ObjectIdentifier(context.runtime)] ?? ""
        let name = component.name
        return AIUIComponentContext(messageID: messageID, isStreaming: context.isStreaming, host: host) { event, label, payload in
            var payload = payload
            payload["component"] = .string(name)
            host.send(AIUIAction(messageID: messageID, format: .openUI, name: event, label: label, payload: payload))
        }
    }

    private func handle(_ event: ActionEvent, messageID: String) {
        guard let host else { return }
        if event.isOpenURL {
            if let url = event.url {
                host.open(url)
            }
            return
        }
        var payload: [String: RawJSON] = [:]
        var params = event.params
        if let context = event.context, !context.isEmpty {
            payload["context"] = .string(context)
        }
        params["context"] = nil
        if !params.isEmpty {
            payload["params"] = .dictionary(params.rawJSON)
        }
        if let form = event.formName {
            payload["form"] = .string(form)
        }
        if let values = event.formState, !values.isEmpty {
            payload["values"] = .dictionary(values.rawJSON)
        }
        let label = event.humanFriendlyMessage.isEmpty ? (event.context ?? event.type) : event.humanFriendlyMessage
        host.send(AIUIAction(messageID: messageID, format: .openUI, name: event.type, label: label, payload: payload))
    }
}

extension ComponentLibrary {
    // The library with each of its parts coming in as it shows, as StreamChatAI brings in the
    // parts of a reply's UI and the app's components.
    var withEntrances: ComponentLibrary {
        ComponentLibrary(
            root: root,
            components: components.map { definition in
                guard let renderer = definition.renderer else { return definition }
                return ComponentDefinition(definition.schema) { context in
                    renderer(context).aiUIEntrance()
                }
            },
            componentGroups: componentGroups,
            defaultPromptOptions: defaultPromptOptions
        )
    }
}

/// A reply's UI, which follows its program as the agent writes it.
struct OpenUIMessageView: View {
    let runtime: OpenUIRuntime
    let program: String
    let isStreaming: Bool

    var body: some View {
        OpenUIView(runtime: runtime)
            .onAppear { runtime.update(response: program, isStreaming: isStreaming) }
            .onChange(of: program) { runtime.update(response: program, isStreaming: isStreaming) }
            .onChange(of: isStreaming) { runtime.update(response: program, isStreaming: isStreaming) }
    }
}
