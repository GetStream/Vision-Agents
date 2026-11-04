import Foundation
import SwiftUI

// Built into one module with StreamChatA2UI's code, whose prompt builder isn't public.
//
// Writes what the agent is told to write replies' UI in A2UI with, in the basic catalog and the
// app's own components. `generate.sh` runs it:
//
//     a2ui-prompt prompt.json > instructions.md
//
// prompt.json, with paths relative to it:
//
//     {
//       "components": "components.json",          portable definitions of the app's components
//       "catalog_id": "getstream.io:app-catalog",  the id of the catalog with them
//       "fence": "ui:a2ui@1.0",                    the info string of a reply's UI block
//       "keep": ["Text", "Column", ...],           the basic catalog's components and functions to
//                                                  describe; all when absent
//       "preamble": "...",                         what the model is told first
//       "notes": ["..."],                          guidance on the app's components
//       "rules": ["..."],                          rules added after the protocol's
//       "examples": ["..."]                        replies with their UI
//     }
//
// Every block of the examples must render with the catalog, or nothing is written: a broken
// example teaches the model to write broken UI.

struct Config: Decodable {
    var components: String?
    var catalogId: String?
    var fence: String?
    var keep: [String]?
    var preamble: String?
    var notes: [String]?
    var rules: [String]?
    var examples: [String]?
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: a2ui-prompt <prompt.json>")
}

let configURL = URL(fileURLWithPath: CommandLine.arguments[1])
let config: Config
do {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    config = try decoder.decode(Config.self, from: Data(contentsOf: configURL))
} catch {
    fail("Couldn't read \(configURL.path): \(error)")
}

var catalog = BasicCatalog.catalog
if let keep = config.keep.map(Set.init) {
    catalog = Catalog(
        catalogId: catalog.catalogId,
        title: catalog.title,
        description: catalog.description,
        instructions: catalog.instructions,
        components: catalog.components.values.filter { keep.contains($0.name) },
        functions: catalog.functions.values.filter { keep.contains($0.name) }
    )
}

var components: [ComponentDefinition] = []
if let path = config.components {
    let url = URL(fileURLWithPath: path, relativeTo: configURL.deletingLastPathComponent())
    do {
        let definitions = Json.array(Json.decode(try Data(contentsOf: url))) ?? []
        components = try definitions.map { definition in
            try PortableComponent.definition(Json.map(definition) ?? [:]) { _ in AnyView(EmptyView()) }
        }
    } catch {
        fail("Couldn't read \(url.path): \(error)")
    }
}
catalog = catalog.adding(components: components, catalogId: config.catalogId ?? "app-catalog")

let fence = config.fence ?? "ui:a2ui@1.0"
var broken: [String] = []
for example in config.examples ?? [] {
    for block in example.components(separatedBy: "```\(fence)\n").dropFirst() {
        let code = block.components(separatedBy: "\n```").first ?? block
        broken += A2UIProgram.problems(in: code, catalog: catalog).map { "example: \($0)" }
    }
}
guard broken.isEmpty else {
    fail(broken.joined(separator: "\n"))
}

let instructions = A2UIInstructions(
    catalog: catalog,
    fence: fence,
    preamble: config.preamble,
    notes: config.notes ?? [],
    rules: config.rules ?? [],
    examples: config.examples ?? []
)
print(instructions.text)
