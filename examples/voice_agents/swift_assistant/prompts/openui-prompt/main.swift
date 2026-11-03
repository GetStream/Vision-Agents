import Foundation
import StreamChatOpenUI

// Writes the system prompt for OpenUI's chat library and the app's own components. `generate.sh`
// runs it:
//
//     openui-prompt prompt.json > instructions.md
//
// prompt.json, with paths relative to it:
//
//     {
//       "components": "components.json",       portable definitions of the app's components
//       "keep": ["Card", "TextContent", ...],   the library's components to describe; all when absent
//       "preamble": "...",                      what the model is told first
//       "notes": ["..."],                       guidance on the app's components
//       "rules": ["..."],                       rules added after the library's
//       "examples": ["..."]                     in place of the library's examples
//     }
//
// Every ```openui-lang block of the examples must parse with the library and the app's
// components, or nothing is written: a broken example teaches the model to write broken UI.

struct Config: Decodable {
    var components: String?
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
    fail("usage: openui-prompt <prompt.json>")
}

let configURL = URL(fileURLWithPath: CommandLine.arguments[1])
let config: Config
do {
    config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
} catch {
    fail("Couldn't read \(configURL.path): \(error)")
}

var library = ComponentLibrary.openUIChat
if let keep = config.keep {
    library = library.removing(Set(library.componentNames).subtracting(keep))
}

if let path = config.components {
    let url = URL(fileURLWithPath: path, relativeTo: configURL.deletingLastPathComponent())
    do {
        let definitions = try OpenUIJSON.parse(Data(contentsOf: url)).arrayValue ?? []
        let components = try definitions.map { OpenUIComponentDefinition(try ComponentSchema(portableDefinition: $0), renderer: nil) }
        library = library.addingContent(components, notes: config.notes ?? [])
    } catch {
        fail("Couldn't read \(url.path): \(error)")
    }
}

var broken: [String] = []
for example in config.examples ?? [] {
    for program in example.components(separatedBy: "```openui-lang\n").dropFirst() {
        let code = program.components(separatedBy: "\n```").first ?? program
        let result = library.parser.parse(code)
        let problems = result.meta.errors.map(\.message)
            + result.meta.unresolved.map { "\($0) is used but not defined" }
            + result.meta.orphaned.map { "\($0) is defined but not used" }
            + (result.root == nil ? ["nothing renders"] : [])
        broken += problems.map { "example: \($0)" }
    }
}
guard broken.isEmpty else {
    fail(broken.joined(separator: "\n"))
}

var options = library.defaultPromptOptions
options.preamble = config.preamble
options.additionalRules += config.rules ?? []
if let examples = config.examples {
    options.examples = examples
}

print(library.systemPrompt(options))
