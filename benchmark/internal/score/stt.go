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

// TranscribeDeepgram sends a mono WAV to Deepgram nova-3.
func TranscribeDeepgram(pcm audio.PCM) (string, error) {
	key := os.Getenv("DEEPGRAM_API_KEY")
	if key == "" {
		return "", fmt.Errorf("score: DEEPGRAM_API_KEY is required")
	}
	var buf bytes.Buffer
	if err := audio.EncodeWAV(&buf, pcm); err != nil {
		return "", err
	}
	req, err := http.NewRequest(http.MethodPost, "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Token "+key)
	req.Header.Set("Content-Type", "audio/wav")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("deepgram HTTP %d: %s", resp.StatusCode, raw)
	}
	var parsed struct {
		Results struct {
			Channels []struct {
				Alternatives []struct {
					Transcript string `json:"transcript"`
				} `json:"alternatives"`
			} `json:"channels"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", err
	}
	if len(parsed.Results.Channels) == 0 || len(parsed.Results.Channels[0].Alternatives) == 0 {
		return "", nil
	}
	return parsed.Results.Channels[0].Alternatives[0].Transcript, nil
}
