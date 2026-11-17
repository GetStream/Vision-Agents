import Foundation

// Built into one module with StreamChatA2UI's code, whose catalog serializer and prompt aren't
// public. `generate.sh` runs it:
//
//     a2ui-catalog            the basic catalog as the renderer serializes it, its guidelines in
//                             `instructions`
//     a2ui-catalog protocol   what the renderer tells a model about A2UI's messages
//
// The Go SDK compiles agent.yaml's `ui:` with both when its format is a2ui.

if CommandLine.arguments.dropFirst().first == "protocol" {
    print(A2UIPromptBuilder.protocolOverview)
    exit(0)
}
let schema = BasicCatalog.catalog.toJsonSchema()
let options: JSONSerialization.WritingOptions = [.sortedKeys, .withoutEscapingSlashes, .prettyPrinted]
let data = try JSONSerialization.data(withJSONObject: schema, options: options)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
