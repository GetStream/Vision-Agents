// Package api holds a transcriber of its own, which nothing meters.
package api

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Listener transcribes behind the router's back.
type Listener struct {
	transcriber stt.STT
}

// Close ends the transcription.
func (l *Listener) Close() error { return l.transcriber.Close() }
