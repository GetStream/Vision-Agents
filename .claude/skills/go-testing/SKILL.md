---
name: go-testing
description: How Go tests are written in acceleration/ and sdks/go. Read before adding a test or a test suite.
---

# Go testing

## Rules

- Group tests in a testify suite (`suite.Suite`). Shared setup lives in the suite, in one place.
- Never mock. A provider is a stub registered in its router's registry, or the real one behind an integration build tag; everything else is real.
- Test behaviour through the outside. Assert on outputs and state, never on which method was called.
- Name a test as the sentence it proves: `TestAnIdAnotherSessionHasIsRefused`.
- Avoid table tests, common as they are in Go. Write one test per behaviour, each named for what it proves, and put what they share in the suite or a helper.
- Anything that needs Postgres, Redis or a provider's API starts with `//go:build integration` and skips, rather than fails, when what it needs is not configured.

## Which kind of test

Read the file for what you are testing:

- [controllers.md](controllers.md): HTTP operations in `internal/api`, run against the whole router with `RouterSuite`. Who may call an endpoint, whose a resource is, fixtures, parallel suites.
- [store.md](store.md): the models and queries in `internal/store`, run against Postgres with `StoreSuite`.
- [ai-routing.md](ai-routing.md): STT, TTS, STS, LLM, search and the other AI providers and their routers. One shared suite per kind of AI that every provider embeds.
