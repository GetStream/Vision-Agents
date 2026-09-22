package api

import (
	"bytes"
	"context"
	"image"
	"os"
	"testing"
)

func TestGeminiImageLive(t *testing.T) {
	if os.Getenv("ATHENA_IMAGE_LIVE") != "1" {
		t.Skip("explicit live image check only")
	}
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		t.Fatal("missing image key")
	}
	data, mime, model, err := NewGeminiImages(key).Generate(context.Background(), "A small yellow ceramic watering can beside a green seedling on a pale cream background. Simple editorial illustration, no text.")
	if err != nil {
		t.Fatal(err)
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width < 1 || model != imageModel || (mime != "image/png" && mime != "image/jpeg") {
		t.Fatalf("invalid image: %v", err)
	}
	if path := os.Getenv("ATHENA_IMAGE_TEST_OUTPUT"); path != "" {
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("generated %s %dx%d, %d bytes", mime, cfg.Width, cfg.Height, len(data))
}
func TestImageEndpointRequiresIdentity(t *testing.T) {
	r, err := (&Server{}).GenerateImage(context.Background(), GenerateImageRequestObject{})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.(GenerateImage401JSONResponse); !ok {
		t.Fatalf("response %T", r)
	}
}
