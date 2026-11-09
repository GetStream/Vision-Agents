package session

import "github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"

// providerMessageBytes bounds provider_message and the other provider texts on a "connector
// call failed" line (dispatcher.record): core.MessageBytes, 1 KiB (AI-1242).
const providerMessageBytes = core.MessageBytes

// providerBody is the start of a 4xx's or a 5xx's body as its log line has it (core.LoggedBody):
// what the transport kept (core.Exchange.ErrorBody) without the UTF-8 sequence its 1 KiB cut may
// end inside, with every http or https URL in it kept to its scheme and host.
func providerBody(body string) string {
	return core.LoggedBody(body)
}
