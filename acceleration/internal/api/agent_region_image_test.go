package api

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// answeringImages is an image router that answers whether a target routes with err.
type answeringImages struct{ err error }

func (r answeringImages) Modality() routing.Modality                    { return routing.Image }
func (r answeringImages) Config() routing.ModalityConfig                { return routing.ModalityConfig{} }
func (r answeringImages) Providers(context.Context) []routing.Candidate { return nil }
func (r answeringImages) Resolve(context.Context, string, []string) ([]routing.Candidate, error) {
	return nil, r.err
}
func (r answeringImages) InRegion(context.Context, string, string, []string) (bool, error) {
	return false, r.err
}

// An image target nothing answers to is the save's mistake, refused with a reason; one the
// router could not look up is the router's, an error rather than a refusal of a good target.
func TestOnlyAnImageTargetNothingAnswersToIsRefused(t *testing.T) {
	config := store.AgentConfig{CustomerID: "acme", Image: "image-fast"}

	unknown := &Server{routers: map[routing.Modality]routing.Inspector{
		routing.Image: answeringImages{err: routing.Refused(errors.New(`routing: unknown target "image-fast"`))},
	}}
	complaint, ok, err := unknown.regionComplaint(t.Context(), config, nil)
	require.NoError(t, err)
	require.False(t, ok)
	require.Contains(t, complaint, "image: image-fast")

	unreachable := &Server{routers: map[routing.Modality]routing.Inspector{
		routing.Image: answeringImages{err: errors.New("routing: health snapshot unavailable")},
	}}
	_, _, err = unreachable.regionComplaint(t.Context(), config, nil)
	require.Error(t, err, "a 5xx for the caller to try again, not a 400 saying the target is wrong")
}
