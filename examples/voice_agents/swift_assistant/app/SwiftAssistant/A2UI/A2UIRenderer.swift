import Foundation
import GenUI
import StreamChatAI
import SwiftUI

/// Shows the replies an agent writes in A2UI 1.0, with GenUI's renderer, and the app's
/// components in the catalog beside the basic catalog's own.
///
/// ```swift
/// let ui = AIUIHost(components: [PlaceCardComponent()]) {
///     A2UIRenderer()
/// } send: { action in
///     try await chat.sendMessage(with: action.label, extraData: action.customData)
/// }
/// ```
///
/// Each reply keeps its own surfaces, so what the person enters in them survives scrolling. A
/// button's event reaches the host as an ``AIUIAction`` named after it, with the person's words,
/// its `userMessage`, as the label, and its surface, its component and its resolved `context` in
/// the payload. Links open on the device.
@MainActor
final class A2UIRenderer: AIUIRenderer {
    /// The version of A2UI it reads.
    static let version = "1.0"
    /// The catalog id the agent names the catalog with the app's components by, unless it is
    /// given another.
    static let defaultCatalogId = "getstream.io:app-catalog"

    let format: AIUIFormat = .a2ui
    /// The catalog the agent writes with, before the app's components join it.
    let catalog: Catalog
    /// The id of the catalog with the app's components.
    let catalogId: String

    /// - Parameters:
    ///   - catalog: The catalog the agent writes with, the basic catalog by default.
    ///   - catalogId: The id the agent names the catalog with the app's components by.
    init(catalog: Catalog = BasicCatalog.catalog, catalogId: String = A2UIRenderer.defaultCatalogId) {
        self.catalog = catalog
        self.catalogId = catalogId
    }

    /// The info string of the fenced block the agent writes a reply's UI in.
    var fence: String { "ui:a2ui@\(Self.version)" }

    func supports(version: String?) -> Bool {
        guard let version else { return true }
        return ["1", "1.0", "1.0.0", "v1.0"].contains(version)
    }

    /// The fence the agent writes its UI in, the catalog's id, and the instructions its model
    /// needs to write UI with the app's components.
    func capabilities(components: [any AIUIComponent]) -> [String: RawJSON] {
        let definitions = components.compactMap { component in
            try? Self.definition(of: component.definition) { _ in AnyView(EmptyView()) }
        }
        let instructions = A2uiInstructions(catalog: catalogs(adding: definitions)[0], fence: fence)
        return [
            "fence": .string(fence),
            "catalog_id": .string(catalogId),
            "instructions": .string(instructions.text)
        ]
    }

    func makeEngine(host: any AIUIHosting) -> some AIUIEngine {
        A2UIEngine(renderer: self, host: host)
    }

    /// The catalog with `components`, under its own id and under the id of the catalog it
    /// extends, so a surface naming either one shows the app's components. With `entrances`, the
    /// catalog's own components come in as they show, as the app's do.
    func catalogs(adding components: [ComponentDefinition], entrances: Bool = false) -> [Catalog] {
        let catalog = entrances ? catalog.withEntrances : catalog
        let extended = catalog.adding(components: components, catalogId: catalogId)
        guard catalog.catalogId != catalogId else { return [extended] }
        return [extended, catalog.adding(components: components)]
    }

    static func definition(of definition: AIUIComponentDefinition, builder: @escaping ComponentViewBuilder) throws -> ComponentDefinition {
        let data = try JSONEncoder().encode(definition)
        guard let portable = Json.decodeMap(String(decoding: data, as: UTF8.self)) else {
            throw PortableComponentError(message: "\(definition.name) has no definition A2UI can read")
        }
        return try PortableComponent.definition(portable, builder: builder)
    }
}

@MainActor
extension Catalog {
    // The catalog with each of its components coming in as it shows, as StreamChatAI brings in
    // the parts of a reply's UI.
    var withEntrances: Catalog {
        Catalog(
            catalogId: catalogId,
            title: title,
            description: description,
            instructions: instructions,
            components: components.values.map(\.withEntrance),
            functions: Array(functions.values)
        )
    }
}

@MainActor
private extension ComponentDefinition {
    var withEntrance: ComponentDefinition {
        let builder = builder
        return ComponentDefinition(
            name: name,
            description: description,
            properties: propertySchema,
            required: requiredProperties,
            allowedParents: allowedParents,
            allowedChildren: allowedChildren,
            examples: examples
        ) { context in
            AnyView(builder(context).aiUIEntrance())
        }
    }
}
