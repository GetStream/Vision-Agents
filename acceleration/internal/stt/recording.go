package stt

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Recording is a whole recording to transcribe, off the live path.
//
// It is a separate contract from STT because the two are not the same job done at
// different speeds. A stream has to guess where a sentence ended before it has heard the
// next one; a recording has the whole thing in front of it, which is why every vendor's
// batch model is both cheaper and more accurate than its streaming one. Nothing is waiting
// to hear the first word, so there is nothing to emit and no session to keep open.
type Recording struct {
	// URL is a fetchable audio or video file. Providers fetch it themselves, which is
	// what makes it the right way to hand over anything long.
	URL string
	// Audio is the file itself, for a caller with a clip and nowhere to host it.
	Audio []byte
	// Languages are the ISO codes spoken, if the caller knows. Empty asks the provider
	// to work it out.
	Languages []string
	// Diarize labels each stretch of speech with who said it, and MaxSpeakers caps how
	// many people it may find. Zero leaves the provider's own limit.
	Diarize     bool
	MaxSpeakers int
	// Words asks for word-level timestamps, which is also what subtitles are rendered
	// from.
	Words bool
	// Format asks for punctuation and the written form of numbers and dates.
	Format bool
	// Redact removes personally identifying information from the transcript.
	Redact bool
	// Summary and Entities ask for the audio intelligence a provider offers on top of the
	// words.
	Summary  bool
	Entities bool
	// Keyterms are the business-specific words the transcriber would otherwise get wrong.
	Keyterms []string
	// Channels transcribes a multichannel recording per channel rather than mixed down.
	Channels int
}

// Validate reports whether there is anything to transcribe.
func (r Recording) Validate() error {
	if r.URL == "" && len(r.Audio) == 0 {
		return errors.New("stt: a recording needs a url or the audio itself")
	}
	if r.URL != "" && len(r.Audio) > 0 {
		return errors.New("stt: a recording is either a url or the audio itself, not both")
	}
	if len(r.Keyterms) > MaxKeyterms {
		return fmt.Errorf("stt: at most %d keyterms are allowed, got %d", MaxKeyterms, len(r.Keyterms))
	}
	return nil
}

// Transcription is a whole recording transcribed.
type Transcription struct {
	// Text is the transcript as prose, which is what most callers want and all of what
	// some do.
	Text string
	// Language is what was spoken, whether it was asked for or detected.
	Language string
	// Words carry the timings, and are present when they were asked for.
	Words []Word
	// Speakers are the people diarization found, in the order they first spoke.
	Speakers []string
	Summary  string
	Entities []Entity
	// AudioDurationMs is how long the recording was, which is what it is billed on.
	AudioDurationMs int64
}

// Word is one word and when it was said.
type Word struct {
	Text       string
	StartMs    int64
	EndMs      int64
	Confidence float64
	// Speaker is who said it, when diarization was asked for.
	Speaker string
}

// Entity is something the recording named.
type Entity struct {
	Type    string
	Text    string
	StartMs int64
	EndMs   int64
}

// Transcriber transcribes a whole recording.
//
// Start and Close are what routing.Provider asks for rather than anything a batch
// transcription needs: an HTTP client has nothing to open, and a provider missing its key
// fails when it is built, which is where the router picks the next candidate.
type Transcriber interface {
	Transcribe(ctx context.Context, recording Recording) (Transcription, error)
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "deepgram".
	Provider() string
	// Model is the model identifier used in stats, e.g. "nova-3".
	Model() string
}

// Subtitles renders a transcript as an SRT or VTT file.
//
// It is done here rather than asked of the provider because the providers that offer it
// offer it inconsistently and the ones that do not offer it have already returned
// everything it takes: a subtitle file is words and timings grouped into lines. Rendering
// it once means a caller gets subtitles from whichever provider served them.
func Subtitles(transcription Transcription, format string) (string, error) {
	switch strings.ToLower(format) {
	case "", "json":
		return "", nil
	case "srt":
		return render(transcription.Words, false), nil
	case "vtt":
		return "WEBVTT\n\n" + render(transcription.Words, true), nil
	default:
		return "", fmt.Errorf("stt: subtitles are srt or vtt, not %q", format)
	}
}

// cueWords is how many words share a subtitle line. Around seven is what a reader gets
// through comfortably in the couple of seconds a cue is on screen.
const cueWords = 7

// render groups words into cues. A recording whose provider returned no timings renders
// as nothing rather than as one unreadable cue over the whole file.
func render(words []Word, web bool) string {
	var built strings.Builder
	for index := 0; index < len(words); index += cueWords {
		cue := words[index:min(index+cueWords, len(words))]

		text := make([]string, 0, len(cue))
		for _, word := range cue {
			text = append(text, word.Text)
		}
		fmt.Fprintf(&built, "%d\n%s --> %s\n%s\n\n",
			index/cueWords+1,
			timecode(cue[0].StartMs, web),
			timecode(cue[len(cue)-1].EndMs, web),
			strings.Join(text, " "))
	}
	return built.String()
}

// timecode writes a position in the two formats the subtitle files disagree about: SRT
// separates the milliseconds with a comma and VTT with a period.
func timecode(ms int64, web bool) string {
	position := time.Duration(ms) * time.Millisecond
	separator := ","
	if web {
		separator = "."
	}
	return fmt.Sprintf("%02d:%02d:%02d%s%03d",
		int64(position.Hours()),
		int64(position.Minutes())%60,
		int64(position.Seconds())%60,
		separator,
		ms%1000)
}
