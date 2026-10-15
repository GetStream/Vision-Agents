package session

import "github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"

// providerMessageBytes bounds provider_message on a "connector call failed" line
// (dispatcher.record): 1 KiB, the ticket's choice (AI-1242), enough for a provider's error
// sentence while a gateway's HTML page cannot fill the log.
const providerMessageBytes = 1 << 10

// providerMessage is what err, the error a failed call returned, says, for its log line: the
// tool's own words for an MCP isError (core.ToolError, whose Error is its Message), the SDK's
// or the JSON-RPC error's text, or the refusal's reason (dispatcher.recheck). Empty for nil.
//
// Every http or https URL in it keeps only its scheme and host, since its userinfo, path or
// query can carry a token. The text is searched rather than a *url.Error unwrapped: the MCP
// SDK also flattens one into text with %v, where errors.As no longer finds it (go-sdk v1.8.0
// mcp/streamable.go:2174 and :2560), and a connector's endpoints are all https
// (internal/connectors/core/AGENTS.md, «Untrusted values cannot steer a request»). It is
// then passed through redact, when it is not nil: the call's core.Exchange.Redact, which takes
// out the last request's path and query and the strings of its credential, since the MCP
// client puts a JSON-RPC error's message, and a tool its isError text, in the error, and a
// provider may echo either there (AI-1244). It is then cut to providerMessageBytes at a UTF-8
// boundary.
//
// Example: `Post "https://crm.example/mcp?access_token=x": EOF` is logged as
// `Post "https://crm.example": EOF`, and `invalid token <the token> for /primary/mcp` as
// `invalid token [redacted] for [redacted]`.
func providerMessage(err error, redact func(string) string) string {
	if err == nil {
		return ""
	}
	text := core.KeepHosts(err.Error())
	if redact != nil {
		text = redact(text)
	}
	return core.CutAtRune(text, providerMessageBytes)
}

// providerBody is the start of a 4xx's or a 5xx's body as its log line has it (core.LoggedBody):
// what the transport kept (core.Exchange.ErrorBody) without the UTF-8 sequence its 1 KiB cut may
// end inside, with every http or https URL in it kept to its scheme and host.
func providerBody(body string) string {
	return core.LoggedBody(body)
}
