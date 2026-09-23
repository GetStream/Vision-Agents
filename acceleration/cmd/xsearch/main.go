// Command xsearch asks Grok what a set of X accounts have been posting about, and prints
// the answer with the posts it came from.
//
// It goes through xAI's Responses API and its server-side x_search tool rather than the X
// API itself, because XAI_API_KEY is the X credential this repository already has. Grok
// does the searching and reading; what comes back is a summary plus citations.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

const (
	apiKeyEnvVar = "XAI_API_KEY"
	endpoint     = "https://api.x.ai/v1/responses"
)

// maxHandles is the limit the x_search tool puts on allowed_x_handles.
const maxHandles = 20

func main() {
	query := flag.String("query", "", "what to ask about the posts (required)")
	handles := flag.String("handles", "", "comma-separated X handles to search, max 20")
	days := flag.Int("days", 30, "how far back to search")
	model := flag.String("model", "grok-4.7", "xAI model doing the searching")
	timeout := flag.Duration("timeout", 3*time.Minute, "how long to wait for the answer")
	flag.Parse()

	if err := run(*query, splitHandles(*handles), *days, *model, *timeout); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(query string, handles []string, days int, model string, timeout time.Duration) error {
	if strings.TrimSpace(query) == "" {
		return fmt.Errorf("-query is required")
	}
	if len(handles) > maxHandles {
		return fmt.Errorf("x_search accepts at most %d handles, got %d", maxHandles, len(handles))
	}
	apiKey, err := apiKey()
	if err != nil {
		return err
	}

	search := map[string]any{"type": "x_search"}
	if len(handles) > 0 {
		search["allowed_x_handles"] = handles
	}
	if days > 0 {
		search["from_date"] = time.Now().AddDate(0, 0, -days).Format(time.DateOnly)
	}
	body, err := json.Marshal(map[string]any{
		"model": model,
		"input": []map[string]string{{"role": "user", "content": query}},
		"tools": []map[string]any{search},
		"store": false,
	})
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("xai: %s: %s", resp.Status, strings.TrimSpace(string(payload)))
	}

	text, sources, err := answer(payload)
	if err != nil {
		return err
	}
	fmt.Println(text)
	if len(sources) > 0 {
		fmt.Println("\nSources:")
		for _, source := range sources {
			fmt.Println("- " + source)
		}
	}
	return nil
}

// response is the part of the Responses API payload this command reads: the text of the
// assistant's message and the posts cited in it.
type response struct {
	Citations []string `json:"citations"`
	Output    []struct {
		Type    string `json:"type"`
		Content []struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Annotations []struct {
				Type string `json:"type"`
				URL  string `json:"url"`
			} `json:"annotations"`
		} `json:"content"`
	} `json:"output"`
}

// answer pulls the message text and the cited URLs out of a response. The annotations on
// the text are the posts the answer actually rests on; the list beside it is everything
// the search touched, profile pages included, so it is only used when there are no
// annotations at all.
func answer(payload []byte) (string, []string, error) {
	var parsed response
	if err := json.Unmarshal(payload, &parsed); err != nil {
		return "", nil, fmt.Errorf("xai: %w", err)
	}

	var text strings.Builder
	var sources []string
	seen := map[string]bool{}
	add := func(url string) {
		if url != "" && !seen[url] {
			seen[url] = true
			sources = append(sources, url)
		}
	}

	for _, item := range parsed.Output {
		if item.Type != "message" {
			continue
		}
		for _, content := range item.Content {
			if content.Type != "output_text" {
				continue
			}
			text.WriteString(content.Text)
			for _, annotation := range content.Annotations {
				if annotation.Type == "url_citation" {
					add(annotation.URL)
				}
			}
		}
	}
	if len(sources) == 0 {
		for _, citation := range parsed.Citations {
			add(citation)
		}
	}

	if text.Len() == 0 {
		return "", nil, fmt.Errorf("xai: no message in response: %s", strings.TrimSpace(string(payload)))
	}
	return text.String(), sources, nil
}

// apiKey takes the key from the environment, which is where a cloud agent has it, and
// falls back to the nearest .env above the working directory, which is where a checkout
// has it. godotenv.Load never overrides what is already set, so the environment wins.
func apiKey() (string, error) {
	if key := os.Getenv(apiKeyEnvVar); key != "" {
		return key, nil
	}
	if path := findDotEnv(); path != "" {
		if err := godotenv.Load(path); err != nil {
			return "", fmt.Errorf("%s: %w", path, err)
		}
	}
	key := os.Getenv(apiKeyEnvVar)
	if key == "" {
		return "", fmt.Errorf("%s is not set, and no .env above the working directory has it", apiKeyEnvVar)
	}
	return key, nil
}

// findDotEnv walks up from the working directory to the first .env, and returns an empty
// string if there is none.
func findDotEnv() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		path := filepath.Join(dir, ".env")
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return path
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// splitHandles turns the flag into handles the tool accepts: no @, no blanks.
func splitHandles(list string) []string {
	var handles []string
	for _, handle := range strings.Split(list, ",") {
		handle = strings.TrimPrefix(strings.TrimSpace(handle), "@")
		if handle != "" {
			handles = append(handles, handle)
		}
	}
	return handles
}
