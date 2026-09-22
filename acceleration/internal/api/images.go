package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"image"
	_ "image/png"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"
)

// ImageGenerator is a server-owned provider; caller input cannot change its URL or credentials.
type ImageGenerator interface {
	Generate(context.Context, string) ([]byte, string, string, error)
}
type FALImages struct {
	key   string
	slots chan struct{}
}

func NewFALImages(key string) *FALImages {
	return &FALImages{key: key, slots: make(chan struct{}, 2)}
}

const imageModel = "alibaba/qwen-image-3/text-to-image"

func (g *FALImages) Generate(ctx context.Context, prompt string) ([]byte, string, string, error) {
	ctx, cancel := context.WithTimeout(ctx, 240*time.Second)
	defer cancel()
	select {
	case g.slots <- struct{}{}:
		defer func() { <-g.slots }()
	case <-ctx.Done():
		return nil, "", "", ctx.Err()
	}
	body, _ := json.Marshal(map[string]any{"prompt": prompt, "image_size": "square_hd", "num_images": 1, "output_format": "png", "sync_mode": true, "enable_safety_checker": true})
	data, err := g.request(ctx, "POST", "https://queue.fal.run/"+imageModel, body)
	if err != nil {
		return nil, "", "", err
	}
	var job struct {
		StatusURL   string `json:"status_url"`
		ResponseURL string `json:"response_url"`
		CancelURL   string `json:"cancel_url"`
	}
	if json.Unmarshal(data, &job) != nil || !falQueueURL(job.StatusURL) || !falQueueURL(job.ResponseURL) || !falQueueURL(job.CancelURL) {
		return nil, "", "", errors.New("invalid image job")
	}
	completed := false
	defer func() {
		if !completed {
			cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			defer cancel()
			_, _ = g.request(cleanup, "PUT", job.CancelURL, nil)
		}
	}()
	timer := time.NewTicker(3 * time.Second)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil, "", "", ctx.Err()
		case <-timer.C:
		}
		data, err = g.request(ctx, "GET", job.StatusURL, nil)
		if err != nil {
			return nil, "", "", err
		}
		var status struct {
			Status string `json:"status"`
		}
		if json.Unmarshal(data, &status) != nil {
			return nil, "", "", errors.New("invalid image status")
		}
		switch status.Status {
		case "COMPLETED":
			completed = true
			data, err = g.request(ctx, "GET", job.ResponseURL, nil)
			if err != nil {
				return nil, "", "", err
			}
			return decodeFALImage(data)
		case "IN_QUEUE", "IN_PROGRESS":
		default:
			return nil, "", "", errors.New("invalid image status")
		}
	}
}

func falQueueURL(raw string) bool {
	u, err := url.Parse(raw)
	return err == nil && u.Scheme == "https" && u.Host == "queue.fal.run" && u.User == nil && strings.HasPrefix(u.Path, "/alibaba/qwen-image-3/") && u.RawQuery == "" && u.Fragment == ""
}

func (g *FALImages) request(ctx context.Context, method, endpoint string, body []byte) ([]byte, error) {
	if !falQueueURL(endpoint) {
		return nil, errors.New("invalid image endpoint")
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Key "+g.key)
	client := http.Client{Timeout: 30 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	res, err := client.Do(req)
	if err != nil {
		return nil, errors.New("image provider unavailable")
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return nil, errors.New("image provider rejected generation")
	}
	data, err := io.ReadAll(io.LimitReader(res.Body, 16<<20+1))
	if err != nil || len(data) > 16<<20 {
		return nil, errors.New("image response too large")
	}
	return data, nil
}

func decodeFALImage(data []byte) ([]byte, string, string, error) {
	var output struct {
		Images []struct {
			URL string `json:"url"`
		} `json:"images"`
	}
	if json.Unmarshal(data, &output) != nil || len(output.Images) != 1 {
		return nil, "", "", errors.New("provider returned no image")
	}
	// sync_mode returns bytes inline; never fetch a provider-supplied URL.
	encoded, ok := strings.CutPrefix(output.Images[0].URL, "data:image/png;base64,")
	if !ok {
		return nil, "", "", errors.New("invalid image response")
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || len(raw) > 10<<20 {
		return nil, "", "", errors.New("invalid image data")
	}
	dimensions, format, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil || format != "png" || dimensions.Width < 1 || dimensions.Height < 1 || dimensions.Width > 2048 || dimensions.Height > 2048 {
		return nil, "", "", errors.New("invalid image dimensions")
	}
	if _, _, err := image.Decode(bytes.NewReader(raw)); err != nil {
		return nil, "", "", errors.New("invalid image data")
	}
	return raw, "image/png", imageModel, nil
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
