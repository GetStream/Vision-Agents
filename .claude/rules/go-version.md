---
paths:
  - "**/*.go"
  - "**/go.mod"
---

# Go Version and Standard Library

## Check go.mod First

Before suggesting Go-specific features or syntax, check the `go` directive in the `go.mod` of the module that owns the file (the nearest one above it). Modules pin different versions, so only use features available in that module's version.

## Prefer Standard Library

Use Go's standard library over third-party packages when feasible. Before suggesting a dependency, check whether the standard library at the module's Go version already provides it, and suggest an external package only when it adds significant value.

## Modern Idioms

When the project's Go version supports newer idioms, prefer them over legacy patterns. If unsure whether a feature exists in the project's Go version, look it up before suggesting.
