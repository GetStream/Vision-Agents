// Package apispec holds what the router's and the usage service's OpenAPI documents share.
package apispec

import "github.com/danielgtaylor/huma/v2"

// DeprecateParams marks a parameter deprecated when its field is tagged `deprecated:"true"`.
// Huma puts that tag on the parameter's schema alone, and a generated client marks a
// parameter deprecated from the parameter's own field
// (https://spec.openapis.org/oas/v3.1.0#parameter-object). Add it to OnAddOperation.
func DeprecateParams(_ *huma.OpenAPI, operation *huma.Operation) {
	for _, param := range operation.Parameters {
		// A parameter declared by hand as a $ref has no schema.
		if param.Schema != nil && param.Schema.Deprecated {
			param.Deprecated = true
		}
	}
}
