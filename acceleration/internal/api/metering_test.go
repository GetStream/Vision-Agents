//go:build integration

package api

import (
	"context"
	"encoding/base64"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
)

// meteredRows waits for the customer's rows of the modality since then to number n, and
// returns them oldest first. They are written behind the request, as every row is.
func (s *RouterSuite) meteredRows(modality routing.Modality, since time.Time, n int) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.ShardDB().NewSelect().Model(&rows).
			Where("customer_id = ? AND modality = ? AND started_at >= ?", s.customerID(), string(modality), since).
			Order("started_at").Scan(context.Background())
		return err == nil && len(rows) >= n
	}, 5*time.Second, 20*time.Millisecond, "%d %s rows were never recorded", n, modality)
	s.Require().Len(rows, n)
	for _, row := range rows {
		s.NotEqual(routing.Unattributed, row.APIKey, "the key that asked pays")
		s.True(row.Success)
	}
	return rows
}

// The knowledge tests take an app of their own, since the standard one's rows are any suite's.

func (s *KnowledgeSuite) TestAKnowledgeWriteIsRecorded() {
	s.useApp(s.data.createApp())
	since := time.Now().UTC()

	s.post("# Pricing\n\nA penny.\n\n# Support\n\nA day.")

	rows := s.meteredRows(routing.Knowledge, since, 1)
	s.Equal("kept", rows[0].Provider)
	s.Equal(metered.KnowledgeWrite, rows[0].Model)
	_, passages := s.knowledge.stored()
	var written int64
	for _, passage := range passages {
		if strings.HasPrefix(passage.Source, s.source) {
			written += passageBytes(passage)
		}
	}
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.BytesWritten: written}, rows[0].Quantities(),
		"one request for both passages, and their bytes as the store billed them")
}

func (s *KnowledgeSuite) TestAKnowledgeDeleteIsRecorded() {
	s.useApp(s.data.createApp())
	s.post("# Refunds\n\nThirty days.")
	listed := s.documents()
	s.Require().Len(listed, 1)
	since := time.Now().UTC()

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete,
		"/v1/agents/knowledge/documents/"+listed[0].Id, nil, nil))

	rows := s.meteredRows(routing.Knowledge, since, 1)
	s.Equal(metered.KnowledgeWrite, rows[0].Model, "a delete is billed as a write")
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.BytesWritten: int64(len(s.source + "#0"))}, rows[0].Quantities())
}

func (s *KnowledgeSuite) TestAFetchIsRecordedAsALookup() {
	s.useApp(s.data.createApp())
	s.post("# Pricing\n\nA penny.\n\n# Support\n\nA day.")
	listed := s.documents()
	s.Require().Len(listed, 1)
	since := time.Now().UTC()

	var passages []KnowledgePassage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents/"+listed[0].Id+"/passages", nil, &passages))
	s.Require().Len(passages, 2)

	rows := s.meteredRows(routing.Knowledge, since, 1)
	s.Equal(metered.KnowledgeLookup, rows[0].Model)
	used := rows[0].Quantities()
	s.EqualValues(1, used[pricing.Requests])
	s.EqualValues(2, used[pricing.Results])
	s.EqualValues(knowledgeQueried, used[pricing.BytesQueried])
	s.Positive(used[pricing.BytesReturned])
}

func (s *VoicesSuite) TestHearingAVoiceIsBilledAndALongLineIsRefused() {
	created := s.createVoice(map[string]any{"name": "founder"})
	s.record(created.Id, map[string]any{
		"audio":    base64.StdEncoding.EncodeToString([]byte("pretend this is speech")),
		"filename": "clip.wav",
	})
	s.prepare(created.Id)
	since := time.Now().UTC()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/voices/"+created.Id+"/preview",
		map[string]any{"provider": "elevenlabs", "text": strings.Repeat("a", voices.MaxLine+1)})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "at most 500 characters")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/preview",
		map[string]any{"provider": "elevenlabs", "text": "Hello there."}, nil))

	rows := s.meteredRows(routing.TTS, since, 1)
	s.Equal("eleven_flash_v2_5", rows[0].Model, "the line is billed as the model it is spoken with")
	s.EqualValues(len("Hello there."), rows[0].Characters)
}

func (s *VoicesSuite) TestCloningAVoiceIsBilledOnce() {
	since := time.Now().UTC()
	created := s.createVoice(map[string]any{"name": "founder"})
	s.record(created.Id, map[string]any{
		"audio":    base64.StdEncoding.EncodeToString([]byte("pretend this is speech")),
		"filename": "clip.wav",
	})

	s.prepare(created.Id)
	s.prepare(created.Id)

	rows := s.meteredRows(routing.TTS, since, 1)
	s.Equal("elevenlabs", rows[0].Provider)
	s.Equal(voices.VoiceClone, rows[0].Model)
	s.Equal(pricing.Quantities{pricing.Clones: 1}, rows[0].Quantities())
	s.Equal(routing.UsageID(voices.VoiceClone, "elevenlabs", s.customerID(), created.Id, "el-cloned"), rows[0].UsageID,
		"named by the clone the provider made, which it handed back both times")
	s.Never(func() bool {
		count, err := s.store.ShardDB().NewSelect().Model((*store.Request)(nil)).
			Where("customer_id = ? AND modality = 'tts'", s.customerID()).Count(context.Background())
		return err != nil || count > 1
	}, 300*time.Millisecond, 50*time.Millisecond, "preparing the same voice again bills nothing")
}
