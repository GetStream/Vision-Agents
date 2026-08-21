package score

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

// TranscriptWord is one timestamped word from STT.
type TranscriptWord struct {
	Text    string `json:"text"`
	StartMS int    `json:"start_ms"`
	EndMS   int    `json:"end_ms"`
}

// Transcript is text plus word timings relative to the beginning of its audio.
type Transcript struct {
	Text  string           `json:"text"`
	Words []TranscriptWord `json:"words,omitempty"`
}

// TranscribeDeepgram sends a mono WAV to Deepgram nova-3.
func TranscribeDeepgram(pcm audio.PCM) (Transcript, error) {
	key := os.Getenv("DEEPGRAM_API_KEY")
	if key == "" {
		return Transcript{}, fmt.Errorf("score: DEEPGRAM_API_KEY is required")
	}
	var buf bytes.Buffer
	if err := audio.EncodeWAV(&buf, pcm); err != nil {
		return Transcript{}, err
	}
	req, err := http.NewRequest(http.MethodPost, "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true", &buf)
	if err != nil {
		return Transcript{}, err
	}
	req.Header.Set("Authorization", "Token "+key)
	req.Header.Set("Content-Type", "audio/wav")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return Transcript{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return Transcript{}, err
	}
	if resp.StatusCode >= 300 {
		return Transcript{}, fmt.Errorf("deepgram HTTP %d: %s", resp.StatusCode, raw)
	}
	var parsed struct {
		Results struct {
			Channels []struct {
				Alternatives []struct {
					Transcript string `json:"transcript"`
					Words      []struct {
						Word           string  `json:"word"`
						PunctuatedWord string  `json:"punctuated_word"`
						Start          float64 `json:"start"`
						End            float64 `json:"end"`
					} `json:"words"`
				} `json:"alternatives"`
			} `json:"channels"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return Transcript{}, err
	}
	if len(parsed.Results.Channels) == 0 || len(parsed.Results.Channels[0].Alternatives) == 0 {
		return Transcript{}, nil
	}
	alternative := parsed.Results.Channels[0].Alternatives[0]
	transcript := Transcript{Text: alternative.Transcript}
	for _, word := range alternative.Words {
		text := word.PunctuatedWord
		if text == "" {
			text = word.Word
		}
		transcript.Words = append(transcript.Words, TranscriptWord{
			Text: text, StartMS: int(word.Start * 1000), EndMS: int(word.End * 1000),
		})
	}
	return transcript, nil
}
