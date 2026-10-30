import Combine
import Foundation
import GenUI
import os
import StreamChatAI
import SwiftUI

/// The A2UI replies of one conversation, each with a message processor of its own.
@MainActor
final class A2UIEngine: AIUIEngine {
    private weak var host: (any AIUIHosting)?
    private let renderer: A2UIRenderer
    private(set) var replies: [String: A2UIReply] = [:]

    init(renderer: A2UIRenderer, host: any AIUIHosting) {
        self.renderer = renderer
        self.host = host
    }

    func view(for document: AIUIDocument, messageID: String) -> A2UIReplyView? {
        guard let source = Self.source(of: document) else { return nil }
        let reply = replies[messageID] ?? makeReply(messageID: messageID, source: source, isStreaming: document.isStreaming)
        return A2UIReplyView(reply: reply, source: source, isStreaming: document.isStreaming)
    }

    // A reply's surfaces start with its messages so far, before any view follows them.
    private func makeReply(messageID: String, source: String, isStreaming: Bool) -> A2UIReply {
        let components = (host?.components ?? []).compactMap { component in
            try? definition(of: component, messageID: messageID)
        }
        let services = RendererServices(openUrl: { [weak host] url in host?.open(url) })
        let processor = A2uiMessageProcessor(
            catalogs: renderer.catalogs(adding: components, entrances: true),
            defaultCatalogId: renderer.catalogId,
            services: services
        )
        let reply = A2UIReply(processor: processor) { [weak self] message in
            self?.handle(message, messageID: messageID)
        }
        replies[messageID] = reply
        reply.update(source: source, isStreaming: isStreaming)
        return reply
    }

    // The A2UI component that shows one of the app's. GenUI's builders return AnyView, so this
    // is where the app's view loses its type, as the catalog's own components do to come in.
    private func definition(of component: some AIUIComponent, messageID: String) throws -> ComponentDefinition {
        try A2UIRenderer.definition(of: component.definition) { [weak self] context in
            AnyView(self?.view(of: component, in: context, messageID: messageID))
        }
    }

    private func view(
        of component: some AIUIComponent,
        in context: ComponentRenderContext,
        messageID: String
    ) -> (some View)? {
        self.context(of: component, messageID: messageID).flatMap { componentContext in
            var arguments: [String: RawJSON] = [:]
            for parameter in component.definition.parameters {
                if let value = context.value(parameter.name) {
                    arguments[parameter.name] = RawJSON(a2ui: value)
                }
            }
            return component.view(arguments: arguments, context: componentContext)
        }
    }

    // What one of the app's components gets besides its arguments, while the host is there.
    private func context(of component: some AIUIComponent, messageID: String) -> AIUIComponentContext? {
        guard let host else { return nil }
        let isStreaming = replies[messageID]?.isStreaming ?? false
        let name = component.name
        return AIUIComponentContext(messageID: messageID, isStreaming: isStreaming, host: host) { event, label, payload in
            var payload = payload
            payload["component"] = .string(name)
            host.send(AIUIAction(messageID: messageID, format: .a2ui, name: event, label: label, payload: payload))
        }
    }

    private func handle(_ message: RendererMessage, messageID: String) {
        switch message {
        case let .action(action):
            let label = action.userMessage ?? buttonText(of: action, messageID: messageID) ?? action.name
            host?.send(AIUIAction(messageID: messageID, format: .a2ui, name: action.name, label: label, payload: [
                "surfaceId": .string(action.surfaceId),
                "sourceComponentId": .string(action.sourceComponentId),
                "context": .dictionary(action.context.mapValues(RawJSON.init(a2ui:)))
            ]))
        case let .error(error):
            logger.error("A reply's A2UI can't show: \(error.code) \(error.message)")
        case let .callAgentFunction(call):
            logger.error("A reply's A2UI calls \(call.callFunction.name), which a chat can't answer")
        case .rendererFunctionResponse:
            break
        }
    }

    // What the button the person tapped says, when its action has no words of its own.
    private func buttonText(of action: RendererAction, messageID: String) -> String? {
        guard let processor = replies[messageID]?.processor,
              let definition = processor.surfaceViewModel(action.surfaceId).definition,
              let childId = definition.component(action.sourceComponentId)?.property("child") as? String,
              let text = definition.component(childId)?.property("text") as? String,
              !text.isEmpty
        else { return nil }
        return text
    }

    // The messages of a document as text: as the agent wrote them, or written out when a backend
    // sent them as JSON.
    private static func source(of document: AIUIDocument) -> String? {
        if let text = document.text {
            return text
        }
        switch document.content {
        case .array, .dictionary:
            guard let data = try? JSONEncoder().encode(document.content) else { return nil }
            return String(decoding: data, as: UTF8.self)
        default:
            return nil
        }
    }
}

/// One reply's surfaces, which follow its messages as the agent writes them.
@MainActor
final class A2UIReply: ObservableObject {
    let processor: A2uiMessageProcessor
    @Published private(set) var surfaceIds: [String] = []
    private(set) var isStreaming = false
    private var applied: [String] = []
    private var messages: AnyCancellable?

    init(processor: A2uiMessageProcessor, onMessage: @escaping @MainActor (RendererMessage) -> Void) {
        self.processor = processor
        messages = processor.rendererMessages.sink { message in
            MainActor.assumeIsolated { onMessage(message) }
        }
    }

    /// Applies the messages of `source` not applied yet. Messages that changed, rather than
    /// grew, start the surfaces again.
    func update(source: String, isStreaming: Bool) {
        self.isStreaming = isStreaming
        let messages = A2uiStream.messages(in: source)
        if !messages.starts(with: applied) {
            processor.clearSurfaces(emitUpdates: false)
            applied = []
        }
        for raw in messages.dropFirst(applied.count) {
            guard var json = Json.decodeMap(raw) else { continue }
            // The fence names the version as 1.0, which a model may copy into the messages.
            if let version = json[A2uiProtocol.versionKey] as? String, ["1", "1.0", "v1"].contains(version) {
                json[A2uiProtocol.versionKey] = A2uiProtocol.version
            }
            do {
                processor.handle(try A2uiMessageDecoder.decode(json))
            } catch {
                logger.error("A reply's A2UI message can't be read: \(String(describing: error))")
            }
        }
        applied = messages
        if surfaceIds != processor.surfaceIds {
            surfaceIds = processor.surfaceIds
        }
    }
}

/// A reply's UI: its surfaces, which follow its messages as the agent writes them.
struct A2UIReplyView: View {
    @ObservedObject var reply: A2UIReply
    let source: String
    let isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(reply.surfaceIds, id: \.self) { surfaceId in
                GenUiSurface(host: reply.processor, surfaceId: surfaceId)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear { reply.update(source: source, isStreaming: isStreaming) }
        .onChange(of: source) { reply.update(source: source, isStreaming: isStreaming) }
        .onChange(of: isStreaming) { reply.update(source: source, isStreaming: isStreaming) }
    }
}

private let logger = Logger(subsystem: "io.getstream.visionagents.SwiftAssistant", category: "A2UI")
