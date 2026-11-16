// Command openapi writes the router's OpenAPI document, rendered from the operations
// declared in internal/api, to api/openapi.yaml. Run it from acceleration/ after changing
// an operation, then regenerate the clients.
package main

import (
	"log"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
)

func main() {
	spec, err := api.Spec()
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("api/openapi.yaml", spec, 0o644); err != nil {
		log.Fatal(err)
	}
}
