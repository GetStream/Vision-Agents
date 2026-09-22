package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"testing"
)

func TestFALImageLive(t *testing.T) {
	if os.Getenv("ATHENA_IMAGE_LIVE") != "1" {
		t.Skip("explicit live image check only")
	}
	key := os.Getenv("FAL_KEY")
	if key == "" {
		t.Fatal("missing image key")
	}
	data, mime, model, err := NewFALImages(key).Generate(context.Background(), "A small yellow ceramic watering can beside a green seedling on a pale cream background. Simple editorial illustration, no text.")
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

func TestFALImageDecoding(t *testing.T) {
	var pngBytes bytes.Buffer
	if err := png.Encode(&pngBytes, image.NewRGBA(image.Rect(0, 0, 8, 8))); err != nil {
		t.Fatal(err)
	}
	payload := func(url string) []byte {
		data, _ := json.Marshal(map[string]any{"images": []any{map[string]string{"url": url}}})
		return data
	}
	valid := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes.Bytes())
	raw, mime, model, err := decodeFALImage(payload(valid))
	if err != nil || !bytes.Equal(raw, pngBytes.Bytes()) || mime != "image/png" || model != "alibaba/qwen-image-3/text-to-image" {
		t.Fatalf("invalid decoded image: %v", err)
	}
	for _, input := range [][]byte{
		[]byte(`{"images":[]}`), []byte(`not json`), payload("https://example.com/image.png"),
		payload("data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(pngBytes.Bytes())),
		payload("data:image/png;base64,broken"),
		payload("data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes.Bytes()[:40])),
	} {
		if _, _, _, err := decodeFALImage(input); err == nil {
			t.Fatal("accepted invalid image response")
		}
	}
}

func TestFALQueueURLStaysOnProvider(t *testing.T) {
	if !falQueueURL("https://queue.fal.run/alibaba/qwen-image-3/requests/test/status") {
		t.Fatal("valid queue URL rejected")
	}
	for _, raw := range []string{"http://queue.fal.run/alibaba/qwen-image-3/test", "https://queue.fal.run.evil.test/alibaba/qwen-image-3/test", "https://user:password@queue.fal.run/alibaba/qwen-image-3/test", "https://queue.fal.run/other/model", "https://queue.fal.run/alibaba/qwen-image-3/test?token=value"} {
		if falQueueURL(raw) {
			t.Fatal("accepted untrusted queue URL")
		}
	}
}
