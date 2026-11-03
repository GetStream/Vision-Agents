package api

import (
	"context"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

var (
	// errNoVoiceNotes is a voice note sent to a deployment that transcribes nothing.
	errNoVoiceNotes = notConfigured("this deployment does not transcribe voice notes")
	// errNothingHeard is a voice note with no words in it: answering it would answer nothing.
	errNothingHeard = invalidRequest("no words were heard in the voice note")
)

// maxBodyWithoutAudio is the most a request with no voice note reads, as it was before voice
// notes: Huma's 1 MiB default. Only a voice note raises it, to maxInlineAudioBody.
const maxBodyWithoutAudio = 1 << 20

// refuseOversizeWithoutAudio refuses a request with no voice note whose size, the bytes of the
// text and sources it carries, is over maxBodyWithoutAudio, with the refusal Huma gives a body
// over its limit. The operations that take a voice note read up to maxInlineAudioBody before
// the handler sees what is in it, so this holds the rest of them to the limit they had.
func refuseOversizeWithoutAudio(audio *AudioSource, size int) error {
	if audio == nil && size > maxBodyWithoutAudio {
		return payloadTooLarge(fmt.Sprintf("request body is too large limit=%d bytes", maxBodyWithoutAudio))
	}
	return nil
}

// withHeardNote is the question of text sent with the voice note at audio: what was typed,
// then what was said, and the note, for a turn to show; text alone when audio is nil.
//
// The note is heard after every check that can refuse the question beforehand, since hearing
// it is billed. Three refusals can still come after it is heard, and the note is billed then:
// the transcript pushes a typed text over the 1 MiB command limit, a second turn starts between
// the check and the start of this one, or the note fails to upload to Chat. A retry of the
// question under commandID carries what its first hearing heard. images is whether photos go
// with it, and asked is what the caller changes about this reply.
func (s *Server) withHeardNote(ctx context.Context, found *session.Session, commandID, text string, audio *AudioSource, images bool, asked options.LLM) (string, []conversation.VoiceNote, error) {
	if audio == nil {
		return text, nil, nil
	}
	heard, err := found.CheckRespond(ctx, commandID, text, images, asked)
	if err != nil {
		return "", nil, refusedResponse(err)
	}
	note, failure := s.hearVoiceNote(ctx, found.Spec(), audio.Url, heard)
	if failure != nil {
		return "", nil, failure
	}
	return withVoiceNote(text, note.Heard), []conversation.VoiceNote{note}, nil
}

// hearVoiceNote fetches the voice note at source, a URL or a data URI, and transcribes it in
// the session's languages for its customer and agent, under its tags and attribution, as the
// session's own model work is. It returns the audio, for a conversation kept in Stream Chat to
// show, and what was said in it. A note heard before, the retry of a question, is fetched to be
// matched on and carries what was heard then rather than being transcribed and billed again.
func (s *Server) hearVoiceNote(ctx context.Context, spec session.Spec, source, heard string) (conversation.VoiceNote, error) {
	if s.streams == nil || s.streams.Transcriptions == nil {
		return conversation.VoiceNote{}, errNoVoiceNotes
	}
	// The person is waiting for an answer, so the note is held to what an inline
	// transcription is.
	ctx, cancel := context.WithTimeout(ctx, inlineDeadline)
	defer cancel()
	note, err := voiceNoteAt(ctx, s.imageHTTP, source)
	if err != nil {
		return conversation.VoiceNote{}, invalidRequest(fmt.Sprintf("the voice note could not be used: %v", err))
	}
	if heard != "" {
		note.Heard = heard
		return note, nil
	}

	held, recording := recordingOf(options.STT{Languages: spec.LanguageHints}, "", note.Audio)
	transcript, _, failure := s.streams.Transcriptions.Transcribe(routing.WithAttribution(ctx, spec.Attribution),
		sttrouter.Recording{CustomerID: spec.CustomerID, Tags: spec.Tags, AgentID: spec.AgentID, CallID: spec.CallID, Region: spec.Region,
			Options: held, Source: recording})
	if answer, refused := moneyError(failure); refused {
		return conversation.VoiceNote{}, answer
	}
	if failure != nil {
		return conversation.VoiceNote{}, modelFailed(failure)
	}
	note.Heard = strings.TrimSpace(transcript.Text)
	if note.Heard == "" {
		return conversation.VoiceNote{}, errNothingHeard
	}
	return note, nil
}

// voiceNoteAt is the audio at source: a data URI's bytes, or what a URL answers, fetched
// with client, at most maxInlineAudio either way.
func voiceNoteAt(ctx context.Context, client *http.Client, source string) (conversation.VoiceNote, error) {
	var data []byte
	var labelled string
	if strings.HasPrefix(source, "data:") {
		parsed, err := llm.ParseDataURI(source)
		if err != nil {
			return conversation.VoiceNote{}, errors.New("it is not a base64 data URI with a type")
		}
		if len(parsed.Data) > maxInlineAudio {
			return conversation.VoiceNote{}, fmt.Errorf("it is over %d MB", maxInlineAudio>>20)
		}
		data, labelled = parsed.Data, parsed.MIME
	} else {
		var err error
		if data, labelled, err = fetchAtMost(ctx, client, source, maxInlineAudio); err != nil {
			return conversation.VoiceNote{}, err
		}
	}
	kind, err := audioKind(data, labelled)
	if err != nil {
		return conversation.VoiceNote{}, err
	}
	return conversation.VoiceNote{MIME: kind, Audio: data}, nil
}

// audioKind is what kind of audio data is, read from its bytes, since a CDN may label an error
// page anything. The containers a voice note is recorded in that the bytes alone call video,
// MP4 and WebM, are taken as their audio. A format the bytes do not name (raw AAC, FLAC) is
// taken as labelled, when the label is an audio type.
func audioKind(data []byte, labelled string) (string, error) {
	switch sniffed := http.DetectContentType(data); {
	case strings.HasPrefix(sniffed, "audio/"):
		return sniffed, nil
	case sniffed == "application/ogg":
		return "audio/ogg", nil
	case sniffed == "video/mp4":
		return "audio/mp4", nil
	case sniffed == "video/webm":
		return "audio/webm", nil
	case sniffed == "application/octet-stream":
		if kind, _, err := mime.ParseMediaType(labelled); err == nil && strings.HasPrefix(kind, "audio/") {
			return kind, nil
		}
		return "", errors.New("it is not audio")
	default:
		return "", fmt.Errorf("it is %s, not audio", sniffed)
	}
}

// withVoiceNote is the text of a question a voice note was sent with: what was typed, then
// what was said.
func withVoiceNote(typed, said string) string {
	if typed == "" {
		return said
	}
	return typed + "\n\n" + said
}
