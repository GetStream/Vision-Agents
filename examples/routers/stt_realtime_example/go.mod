module github.com/GetStream/Vision-Agents/examples/routers/stt_realtime_example

go 1.25.0

require (
	github.com/GetStream/Vision-Agents/sdks/go v0.0.0
	github.com/joho/godotenv v1.5.1
)

require (
	github.com/apapsch/go-jsonmerge/v2 v2.0.0 // indirect
	github.com/golang-jwt/jwt/v5 v5.3.1 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/oapi-codegen/runtime v1.6.0 // indirect
	golang.org/x/net v0.58.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

// A path replacement as well as a workspace entry, so that running this with GOWORK=off, or
// from a copy of the directory taken out of the repo, still points at the SDK next door.
replace github.com/GetStream/Vision-Agents/sdks/go => ../../../sdks/go
