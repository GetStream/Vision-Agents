module github.com/GetStream/Vision-Agents/examples/voice_agents/swift_demo/configure

go 1.25.0

require github.com/GetStream/Vision-Agents/sdks/go v0.0.0

require (
	github.com/GetStream/getstream-go/v5 v5.2.0 // indirect
	github.com/apapsch/go-jsonmerge/v2 v2.0.0 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/golang-jwt/jwt/v5 v5.3.1 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/kr/pretty v0.3.1 // indirect
	github.com/oapi-codegen/runtime v1.6.0 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	golang.org/x/net v0.57.0 // indirect
	gopkg.in/check.v1 v1.0.0-20201130134442-10cb98267c6c // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

// A path replacement as well as a workspace entry, so that running this with GOWORK=off, or
// from a copy of the directory taken out of the repo, still points at the SDK next door.
replace github.com/GetStream/Vision-Agents/sdks/go => ../../../../sdks/go
