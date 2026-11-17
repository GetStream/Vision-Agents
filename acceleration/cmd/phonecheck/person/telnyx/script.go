package telnyx

import (
	"fmt"
	"time"
)

const markerSentence = "This is a phonecheck test call."

// dialScript is what the person says on an inbound call: a short pause, the marker, then
// silence for the talk time so the router's session stays up. TeXML takes the same verbs
// as TwiML.
func dialScript(talk time.Duration) string {
	return fmt.Sprintf(`<Response><Pause length="5"/><Say>%s</Say><Pause length="%d"/><Hangup/></Response>`,
		markerSentence, int(talk.Seconds()))
}
