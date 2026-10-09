package agent

import (
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/audioturn"
	"github.com/stretchr/testify/require"
)

func TestPCM16LERingWrapsChronologicallyAndCopiesSnapshots(t *testing.T) {
	ring := &pcm16leRing{sample: make([]int16, eotMinSamples)}
	first := make([]int16, eotMinSamples)
	for i := range first {
		first[i] = int16(i)
	}
	ring.append(first)
	second := []int16{1000, 1001}
	ring.append(second)
	snapshot := ring.snapshot()
	require.Len(t, snapshot, eotMinSamples*2)
	require.Equal(t, int16(2), int16(uint16(snapshot[0])|uint16(snapshot[1])<<8))
	require.Equal(t, int16(1000), int16(uint16(snapshot[len(snapshot)-4])|uint16(snapshot[len(snapshot)-3])<<8))
	require.Equal(t, int16(1001), int16(uint16(snapshot[len(snapshot)-2])|uint16(snapshot[len(snapshot)-1])<<8))
	before := append([]byte(nil), snapshot...)
	ring.append([]int16{2000})
	require.Equal(t, before, snapshot)
	ring.clear()
	require.Nil(t, ring.snapshot())
}

func eotJSON(id string, samples int, probability float64) string {
	return fmt.Sprintf(`{"request_id":%q,"model":%q,"release":%q,"probability":%v,"wait_probability":%v,"sample_rate":16000,"samples":%d,"window_samples":%d}`,
		id, audioturn.DefaultModel, audioturn.Release, probability, 1-probability, samples, samples)
}

func writeEOTResponse(t *testing.T, w http.ResponseWriter, id string, samples int, probability float64) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if _, err := io.WriteString(w, eotJSON(id, samples, probability)); err != nil {
		t.Error(err)
	}
}

func TestRetainEOTAudioKeepsZeroSamplesAndRejectsWrongFormat(t *testing.T) {
	a := &Agent{
		options:      Options{EOT: &audioturn.Client{}},
		audioHistory: make(map[string]*pcm16leRing),
	}
	samples := make([]int16, eotMinSamples)
	for i := range samples {
		if i%4 == 0 {
			samples[i] = 1234
		}
	}
	a.retainEOTAudio("participant", eotSampleRate, 1, samples)
	a.retainEOTAudio("participant", eotSampleRate/2, 1, samples)
	a.retainEOTAudio("participant", eotSampleRate, 2, samples)

	got := a.eotAudioSnapshot("participant")
	require.Len(t, got, len(samples)*2)
	require.Equal(t, samples, decodeEOTPCM(got), "zero-valued frames remain in the scoring window; no VAD filtering is applied")
}

func decodeEOTPCM(pcm []byte) []int16 {
	if len(pcm)%2 != 0 {
		panic("odd PCM byte length")
	}
	samples := make([]int16, len(pcm)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
	}
	return samples
}
