package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

// ImageGenerator is a server-owned provider; caller input cannot change its URL or credentials.
type ImageGenerator interface {
	Generate(context.Context, string) ([]byte, string, string, error)
}
type GeminiImages struct {
	key   string
	slots chan struct{}
}

func NewGeminiImages(key string) *GeminiImages {
	return &GeminiImages{key: key, slots: make(chan struct{}, 2)}
}

const imageModel = "gemini-3.1-flash-image"

func (g *GeminiImages) Generate(ctx context.Context, prompt string) ([]byte, string, string, error) {
	ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	select {
	case g.slots <- struct{}{}:
		defer func() { <-g.slots }()
	case <-ctx.Done():
		return nil, "", "", ctx.Err()
	}
	body, _ := json.Marshal(map[string]any{"contents": []any{map[string]any{"parts": []any{map[string]string{"text": prompt}}}}, "generationConfig": map[string]any{"responseModalities": []string{"IMAGE"}, "imageConfig": map[string]string{"imageSize": "1K", "aspectRatio": "1:1"}}})
	req, err := http.NewRequestWithContext(ctx, "POST", "https://generativelanguage.googleapis.com/v1beta/models/"+imageModel+":generateContent", bytes.NewReader(body))
	if err != nil {
		return nil, "", "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", g.key)
	client := http.Client{Timeout: 90 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	res, err := client.Do(req)
	if err != nil {
		return nil, "", "", errors.New("image provider unavailable")
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, "", "", errors.New("image provider rejected generation")
	}
	data, err := io.ReadAll(io.LimitReader(res.Body, 16<<20+1))
	if err != nil || len(data) > 16<<20 {
		return nil, "", "", errors.New("image response too large")
	}
	var output struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					InlineData *struct {
						MimeType string `json:"mimeType"`
						Data     string `json:"data"`
					} `json:"inlineData"`
					Thought bool `json:"thought"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if json.Unmarshal(data, &output) != nil {
		return nil, "", "", errors.New("invalid image response")
	}
	for _, candidate := range output.Candidates {
		for _, part := range candidate.Content.Parts {
			if part.Thought || part.InlineData == nil {
				continue
			}
			p := part.InlineData
			if p.MimeType != "image/png" && p.MimeType != "image/jpeg" {
				continue
			}
			raw, err := base64.StdEncoding.DecodeString(p.Data)
			if err != nil || len(raw) > 10<<20 {
				continue
			}
			dimensions, format, err := image.DecodeConfig(bytes.NewReader(raw))
			if err != nil || dimensions.Width < 1 || dimensions.Height < 1 || dimensions.Width > 4096 || dimensions.Height > 4096 {
				continue
			}
			if (format == "png" && p.MimeType != "image/png") || (format == "jpeg" && p.MimeType != "image/jpeg") {
				continue
			}
			return raw, p.MimeType, imageModel, nil
		}
	}
	return nil, "", "", errors.New("provider returned no image")
}
func (s *Server) GenerateImage(ctx context.Context, request GenerateImageRequestObject) (GenerateImageResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return GenerateImage401JSONResponse{missingCustomer()}, nil
	}
	if s.images == nil {
		return GenerateImage404JSONResponse{NotFoundJSONResponse{Error: "image generation unavailable"}}, nil
	}
	if request.Body == nil || strings.TrimSpace(request.Body.Prompt) == "" || !utf8.ValidString(request.Body.Prompt) || utf8.RuneCountInString(request.Body.Prompt) > 4000 {
		return GenerateImage400JSONResponse{badRequest("prompt must contain 1 to 4000 characters")}, nil
	}
	data, mime, model, err := s.images.Generate(ctx, request.Body.Prompt)
	if err != nil {
		return GenerateImage502JSONResponse{Error: "image generation failed"}, nil
	}
	return GenerateImage200JSONResponse{Data: base64.StdEncoding.EncodeToString(data), MediaType: mime, Model: model}, nil
}
