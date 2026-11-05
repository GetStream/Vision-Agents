// Command transcribe_only streams a recorded call through the router's realtime STT to
// chorus-asr alone, and prints what it wrote.
//
// Pick this when you want chorus-asr's words and nothing else: your own pipeline decides when
// to answer, and the router asks the server only for words (POST /v1/transcribe).
//
// The router runs with ROUTER_CHORUS_ASR_URL; ROUTER_CHORUS_ASR_MODE stays off (../README.md).
//
//	go run .
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/joho/godotenv"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// audio is the realtime STT example's clip, sent a chunk at a time as a call arrives.
const audio = "../../stt_realtime_example/saturday_seven_thirty.wav"

// settling outlasts the 2.5 s pause at which chorus-asr ends an utterance.
const settling = 3 * time.Second

func main() {
	ctx := context.Background()
	_ = godotenv.Load("../../../../.env")

	client, err := stream.NewClient(stream.Backend{})
	if err != nil {
		log.Fatal(err)
	}
	// routers/chorus-asr.yaml names stream/chorus-asr and nothing else.
	if _, err := client.SyncRouters(ctx, "../routers"); err != nil {
		log.Fatal(err)
	}
	transcriber, err := client.Router("chorus-asr").STT().Realtime(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	call, err := stream.RecordedCall(audio)
	if err != nil {
		log.Fatal(err)
	}

	go func() {
		for heard := range transcriber.Transcripts() {
			if heard.Final || heard.Error != "" {
				fmt.Printf("%s/%s: %s%s\n", heard.Provider, heard.Model, heard.Text, heard.Error)
			}
		}
	}()
	for chunk := range call {
		if err := transcriber.Send(chunk); err != nil {
			log.Fatal(err)
		}
	}
	time.Sleep(settling)
	if err := transcriber.Close(); err != nil {
		log.Fatal(err)
	}
}
