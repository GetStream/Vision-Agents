package synth

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
)

const defaultVoice = "VR6AewLTigWG4xSOukaG"

func cacheDir(root string) string {
	return filepath.Join(root, "cache", "tts")
}

func hash(voice, text string) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%s\x00%d", voice, text, audio.Rate)))
	return fmt.Sprintf("%x", sum[:12])
}

func pathFor(root, voice, text string) string {
	return filepath.Join(cacheDir(root), hash(voice, text)+".wav")
}

// LoadOrSynth returns 16 kHz PCM for text, synthesizing via ElevenLabs on a miss.
func LoadOrSynth(root, voice, text string) ([]int16, error) {
	if voice == "" {
		voice = os.Getenv("ELEVENLABS_VOICE_ID")
	}
	if voice == "" {
		voice = defaultVoice
	}
	path := pathFor(root, voice, text)
	if pcm, err := audio.ReadWAV(path); err == nil && pcm.Rate == audio.Rate {
		return pcm.Samples, nil
	}
	pcm, err := elevenLabs(voice, text)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := audio.WriteWAV(path, pcm); err != nil {
		return nil, err
	}
	return pcm.Samples, nil
}

// Pack renders every speech turn in a scenario pack.
func Pack(root, voice string, scenarios []scenario.Scenario) error {
	for _, sc := range scenarios {
		for _, text := range sc.SpeechTexts() {
			if _, err := LoadOrSynth(root, voice, text); err != nil {
				return fmt.Errorf("synth %s: %w", sc.ID, err)
			}
		}
	}
	return nil
}

func elevenLabs(voice, text string) (audio.PCM, error) {
	key := os.Getenv("ELEVENLABS_API_KEY")
	if key == "" {
		return audio.PCM{}, fmt.Errorf("synth: ELEVENLABS_API_KEY is required")
	}
	url := fmt.Sprintf("https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=pcm_16000", voice)
	body, err := json.Marshal(map[string]string{
		"text":     text,
		"model_id": "eleven_flash_v2_5",
	})
	if err != nil {
		return audio.PCM{}, err
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return audio.PCM{}, err
	}
	req.Header.Set("xi-api-key", key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/octet-stream")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return audio.PCM{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return audio.PCM{}, err
	}
	if resp.StatusCode >= 300 {
		return audio.PCM{}, fmt.Errorf("elevenlabs HTTP %d: %s", resp.StatusCode, raw)
	}
	return audio.FromPCM16LE(raw, audio.Rate), nil
}
