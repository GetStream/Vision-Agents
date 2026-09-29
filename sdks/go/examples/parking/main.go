// Command parking counts the cars in a parking lot and says how full it is.
//
// The picture goes through the acceleration router to NVIDIA's LocateAnything-3B on
// Baseten (see acceleration/deploy/locate-anything), which is asked for a box around
// every parked car and then around every empty space. The count and the percentage come
// from those boxes, and the boxes are drawn onto a copy of the picture.
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=acme \
//	go run ./examples/parking -image lot.jpg -spaces 60
//
// With -every, the picture is fetched and counted again on that interval, which is what a
// camera publishing a snapshot URL wants.
package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// sample is an aerial photo of a nearly full lot in Abuja, by Wikimedia Commons user
// Aty Jorbes, CC BY-SA 4.0.
const sample = "https://upload.wikimedia.org/wikipedia/commons/d/df/Aerial_view_of_an_Abuja_parking_lot.jpg"

func main() {
	source := flag.String("image", sample, "a picture of the lot: a file or an http URL")
	spaces := flag.Int("spaces", 0, "how many spaces the lot has; 0 counts the empty ones it can see")
	out := flag.String("out", "parking.png", "where the annotated picture is written")
	target := flag.String("target", "locateanything/LocateAnything-3B", "the model to route to")
	every := flag.Duration("every", 0, "count again on this interval; 0 counts once")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *source, *spaces, *out, *target, *every); err != nil && !errors.Is(err, context.Canceled) {
		log.Fatal(err)
	}
}

func run(ctx context.Context, source string, spaces int, out, target string, every time.Duration) error {
	model, err := stream.Router{}.LLM().Realtime(ctx, &acceleration.LlmOptions{Target: &target})
	if err != nil {
		return err
	}
	defer model.Close()

	for {
		if err := count(ctx, model, source, spaces, out); err != nil {
			return err
		}
		if every == 0 {
			return nil
		}
		select {
		case <-time.After(every):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// count fetches the picture once, asks for the boxes, and reports the lot.
func count(ctx context.Context, model *stream.Model, source string, spaces int, out string) error {
	contents, err := load(ctx, source)
	if err != nil {
		return err
	}
	picture, _, err := image.Decode(bytes.NewReader(contents))
	if err != nil {
		return fmt.Errorf("decoding %s: %w", source, err)
	}

	started := time.Now()
	var detections []Detection
	for _, label := range []string{carLabel, freeLabel} {
		answer, err := ask(ctx, model, contents, query(label))
		if err != nil {
			return err
		}
		detections = append(detections, parse(answer, label, picture.Bounds())...)
	}
	took := time.Since(started)

	detections = plausible(detections)
	lot := occupancy(detections, spaces)
	if err := save(out, annotate(picture, detections, lot)); err != nil {
		return err
	}

	fmt.Printf("%s  %d of %d spaces taken: %.0f%% full, %.0f%% free (%d free)  [%s, %s]\n",
		time.Now().Format("15:04:05"), lot.Cars, lot.Spaces, lot.FullPercent(), lot.FreePercent(), lot.Free,
		took.Round(10*time.Millisecond), out)
	return nil
}

// ask sends the picture and the question, and waits for the whole answer.
func ask(ctx context.Context, model *stream.Model, picture []byte, question string) (string, error) {
	err := model.Ask(stream.Question{Messages: []stream.Said{{
		Role: "user",
		Parts: []stream.ContentPart{
			{Image: &stream.Image{Data: picture, MIME: http.DetectContentType(picture)}},
			{Text: question},
		},
	}}})
	if err != nil {
		// A router that refused the socket said why before closing it.
		for answer := range model.Answers() {
			if answer.Error != "" {
				return "", fmt.Errorf("the router: %s", answer.Error)
			}
		}
		return "", err
	}

	for {
		select {
		case answer, open := <-model.Answers():
			if !open {
				return "", errors.New("the router closed the socket before answering")
			}
			if answer.Error != "" {
				return "", fmt.Errorf("the router: %s", answer.Error)
			}
			if answer.Done {
				return answer.Text, nil
			}
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
}

// load reads the picture from a file or fetches it from a URL.
func load(ctx context.Context, source string) ([]byte, error) {
	if !strings.HasPrefix(source, "http://") && !strings.HasPrefix(source, "https://") {
		return os.ReadFile(source)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, source, nil)
	if err != nil {
		return nil, err
	}
	// Wikimedia, where the sample lives, refuses a request that names no user agent.
	request.Header.Set("User-Agent", "vision-agents-parking-example/1.0")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", source, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: %s", source, response.Status)
	}
	return io.ReadAll(response.Body)
}

func save(path string, picture image.Image) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(file, picture); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}
