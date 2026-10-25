// Package crossed is a speech-to-speech vendor that builds a transcriber of its own, which no
// router meters.
package crossed

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/deepgram"
)

// listener is built here rather than by the transcription router's registry.
var listener, _ = deepgram.New(deepgram.Options{})
