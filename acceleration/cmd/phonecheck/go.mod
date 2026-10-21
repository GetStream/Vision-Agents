module github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck

go 1.27.0

require (
	github.com/GetStream/Vision-Agents/acceleration v0.0.0-00010101000000-000000000000
	github.com/GetStream/Vision-Agents/sdks/go v0.0.0
	github.com/stretchr/testify v1.12.1
)

require (
	github.com/GetStream/getstream-go/v5 v5.2.0 // indirect
	github.com/apapsch/go-jsonmerge/v2 v2.0.0 // indirect
	github.com/golang-jwt/jwt/v5 v5.3.1 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/oapi-codegen/runtime v1.6.0 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/net v0.58.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/GetStream/Vision-Agents/acceleration => ../..

replace github.com/GetStream/Vision-Agents/sdks/go => ../../../sdks/go
