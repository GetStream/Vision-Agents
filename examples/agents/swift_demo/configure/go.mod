module github.com/GetStream/Vision-Agents/examples/agents/swift_demo/configure

go 1.25.0

require github.com/GetStream/Vision-Agents/agents-core-go v0.0.0

// A path replacement as well as a workspace entry, so that running this with GOWORK=off, or
// from a copy of the directory taken out of the repo, still points at the SDK next door.
replace github.com/GetStream/Vision-Agents/agents-core-go => ../../../../agents-core-go
