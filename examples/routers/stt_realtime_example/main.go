// Command stt_realtime_example stores two router configs, then transcribes through one.
//
// The two in routers/ differ in one line. stt-realtime-fast asks not to be told who spoke
// and keeps the four models it names as a fallback chain; stt-realtime-accurate asks to be
// told, which narrows the same shape of chain to the one model that can. Neither says
// anything at the call site - Realtime is handed nil options - so which model answers is
// the config's decision, and it is printed with each transcript.
//
// The clip next to this file is streamed a chunk at a time, the way a call arrives.
//
//	go run .                                    # stt-realtime-fast
//	go run . -use-case stt-realtime-accurate
//	go run . -staging
//
// Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
// -staging is the hosted one, which authenticates a Stream app rather than naming a
// customer, so it needs STREAM_API_KEY and STREAM_API_SECRET instead.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/joho/godotenv"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

const staging = "https://accelerate.gcp.stream-io-api.com"

const audio = "saturday_seven_thirty.wav"

// settling is how long the transcripts of the trailing silence are waited for.
const settling = 3 * time.Second

func main() {
	useCase := flag.String("use-case", "stt-realtime-fast",
		"the config to transcribe under: stt-realtime-fast or stt-realtime-accurate")
	hosted := flag.Bool("staging", false, "transcribe on the hosted router rather than a local one")
	flag.Parse()

	if err := run(context.Background(), *useCase, *hosted); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, useCase string, hosted bool) error {
	_ = godotenv.Load("../../../.env")

	// Where the router is and who is calling it, settled once: a local router reads both
	// from the environment, the hosted one authenticates a Stream app instead. Everything
	// below reads the same either way, which is the point of holding a client.
	backend := stream.Backend{}
	if hosted {
		backend = stream.Backend{URL: staging, Authenticate: true}
	}
	client, err := stream.NewClient(backend)
	if err != nil {
		return err
	}

	// Naming a config is not enough on its own: the router has to have been told what the
	// name means, and editing a file in routers/ edits the config rather than adding one.
	if _, err := client.SyncRouters(ctx, "routers"); err != nil {
		return err
	}

	call, err := stream.RecordedCall(audio)
	if err != nil {
		return err
	}

	transcriber, err := client.Router(useCase).STT().Realtime(ctx, nil)
	if err != nil {
		return err
	}

	// Read while writing: the first words are transcribed while the rest of the clip is
	// still going up, and a socket nobody is reading eventually blocks.
	go report(transcriber.Transcripts())

	for chunk := range call {
		if err := transcriber.Send(chunk); err != nil {
			return err
		}
	}

	// The last words of a turn are transcribed after they are spoken, so the tail of the
	// clip settles after the audio has stopped rather than with it.
	time.Sleep(settling)
	return transcriber.Close()
}

// report prints the turns the router settled on.
//
// Most of what arrives before then is the provider's current guess, superseded by the next
// one. The provider and model on each are what the config left open.
func report(transcripts <-chan stream.Transcript) {
	for heard := range transcripts {
		switch {
		case heard.Error != "":
			fmt.Printf("\nerror: %s\n", heard.Error)
		case heard.Final:
			fmt.Printf("\n%s/%s: %s%s\n", heard.Provider, heard.Model, said(heard), heard.Text)
		}
	}
}

// said is who the provider heard, for the models that tell one voice at a microphone from
// another. The rest say nothing, which is not the same as one speaker.
func said(heard stream.Transcript) string {
	if heard.Speaker == "" {
		return ""
	}
	return "speaker " + heard.Speaker + ": "
}
