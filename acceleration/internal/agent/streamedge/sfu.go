package streamedge

import (
	"net/url"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// sfuCheckEvery is how often the SFU is read for a move. The SDK fires nothing when it
// changes, and handlers registered on its signal client are lost when a migration replaces
// that client, so it is polled.
var sfuCheckEvery = 5 * time.Second

// sfuOf names the SFU a call state points at. Only the edge name and the URL's host are
// kept: the URL carries a token.
func sfuOf(state *rtc.CallState) agent.SFU {
	if state == nil {
		return agent.SFU{}
	}
	var host string
	if parsed, err := url.Parse(state.Url); err == nil {
		host = parsed.Hostname()
	}
	return agent.SFU{Edge: state.EdgeName, Host: host}
}

// watchSFU calls changed each time read returns a different SFU than before, until stop closes.
func watchSFU(read func() agent.SFU, every time.Duration, stop <-chan struct{}, changed func(was, now agent.SFU)) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	was := read()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			if now := read(); now != was {
				changed(was, now)
				was = now
			}
		}
	}
}
