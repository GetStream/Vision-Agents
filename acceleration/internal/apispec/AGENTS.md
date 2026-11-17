# internal/apispec

What the router's (`internal/api`) and the usage service's (`internal/usage/server`) OpenAPI documents share: `DeprecateParams`, an `OnAddOperation` hook that marks a parameter tagged `deprecated:"true"` deprecated on the parameter itself, where client generators read it.
