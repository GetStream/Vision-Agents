package caller

import (
	"context"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

func TestPlayLoopback(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate, 220, 10000))
		time.Sleep(200 * time.Millisecond)
		for range loop.AgentRecv {
		}
	}()

	text := "hello table for four"
	eng := Engine{
		Audio: map[string][]int16{
			text: audio.Tone(audio.Rate/2, 180, 10000),
		},
	}
	sc := scenario.Scenario{
		ID:   "t",
		Pack: "restaurant",
		Turns: []scenario.Turn{
			{
				ID:   "intro",
				Text: text,
				Trigger: scenario.Trigger{
					Kind:    scenario.TriggerAfterAgent,
					DelayMS: 50,
				},
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Caller) == 0 {
		t.Fatal("expected caller audio")
	}
	if len(got.Events) != 1 {
		t.Fatalf("events %d", len(got.Events))
	}
}

func TestMidReplyPauseDoesNotAdvanceTurn(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate/5, 220, 10000))
		heard := false
		for frame := range loop.AgentRecv {
			if heard || audio.FrameEnergy(frame.PCM) < audio.DefaultSpeechThreshold {
				continue
			}
			heard = true
			_ = loop.SendAgent(audio.Tone(audio.Rate/5, 210, 10000))
			time.Sleep(400 * time.Millisecond)
			_ = loop.SendAgent(audio.Tone(audio.Rate, 210, 10000))
		}
	}()

	first := "first turn"
	second := "second turn"
	eng := Engine{
		Audio: map[string][]int16{
			first:  audio.Tone(audio.Rate/4, 180, 10000),
			second: audio.Tone(audio.Rate/4, 190, 10000),
		},
		TurnHangoverMS: 1200,
	}
	sc := scenario.Scenario{
		ID:   "pause",
		Pack: "restaurant",
		Turns: []scenario.Turn{
			{ID: "a", Text: first, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
			{ID: "b", Text: second, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Events) != 2 {
		t.Fatalf("events %d", len(got.Events))
	}
	gap := got.Events[1].RecStartMs - got.Events[0].RecEndMs
	if gap < 1000 {
		t.Fatalf("second turn started %dms after first in the recording; pause should have been ignored", gap)
	}
}

func TestDuringAgentOverlapPlaysWhileAgentSpeaks(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(2*audio.Rate, 220, 10000))
		for range loop.AgentRecv {
		}
	}()

	eng := Engine{}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{
				ID:           "cough",
				OverlapSound: "cough",
				Trigger: scenario.Trigger{
					Kind:    scenario.TriggerDuringAgent,
					AfterMS: 200,
				},
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Events) != 1 {
		t.Fatalf("events %d", len(got.Events))
	}
	spans := audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) == 0 {
		t.Fatal("expected agent speech")
	}
	start := got.Events[0].RecStartMs
	inside := false
	for _, s := range spans {
		if start >= s.StartMs && start <= s.EndMs {
			inside = true
			break
		}
	}
	if !inside {
		t.Fatalf("cough RecStartMs=%d not inside agent spans %+v", start, spans)
	}
}

func TestBargeInWaitsForAgentSpeech(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		time.Sleep(1500 * time.Millisecond)
		_ = loop.SendAgent(audio.Tone(2*audio.Rate, 220, 10000))
		for range loop.AgentRecv {
		}
	}()

	text := "wait make it six"
	eng := Engine{
		Audio: map[string][]int16{
			text: audio.Tone(audio.Rate/4, 180, 10000),
		},
	}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{
				ID:   "change",
				Text: text,
				Trigger: scenario.Trigger{
					Kind:    scenario.TriggerBargeIn,
					AfterMS: 200,
				},
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Events) != 1 {
		t.Fatalf("events %d", len(got.Events))
	}
	spans := audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) == 0 {
		t.Fatal("expected agent speech")
	}
	start := got.Events[0].RecStartMs
	inside := false
	for _, s := range spans {
		if start >= s.StartMs && start <= s.EndMs {
			inside = true
			break
		}
	}
	if !inside {
		t.Fatalf("barge RecStartMs=%d not inside agent spans %+v", start, spans)
	}
}

func TestFinalWaitCapturesLateReply(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate/5, 220, 10000))
		heard := false
		for frame := range loop.AgentRecv {
			if heard || audio.FrameEnergy(frame.PCM) < audio.DefaultSpeechThreshold {
				continue
			}
			heard = true
			time.Sleep(2 * time.Second)
			_ = loop.SendAgent(audio.Tone(audio.Rate, 260, 12000))
		}
	}()

	text := "book me"
	eng := Engine{
		Audio: map[string][]int16{
			text: audio.Tone(audio.Rate/4, 180, 10000),
		},
	}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{ID: "intro", Text: text, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	spans := audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) < 2 {
		t.Fatalf("expected greeting and late reply, spans=%d", len(spans))
	}
}
