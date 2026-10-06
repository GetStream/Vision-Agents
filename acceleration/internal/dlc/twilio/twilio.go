// Package twilio is where Twilio's A2P 10DLC registration will go. Twilio registers a
// brand through Trust Hub customer profiles and a campaign as a Messaging Service's
// us_app_to_person resource, which is a different shape enough that it is not built until
// an app on Twilio numbers needs it.
package twilio

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ErrNotImplemented is every answer this registrar gives.
var ErrNotImplemented = errors.New("twilio: 10DLC registration is not implemented yet")

// Registrar is Twilio. It satisfies dlc.Registrar and registers nothing.
type Registrar struct{}

func (Registrar) Name() string { return "twilio" }

func (Registrar) RegisterBrand(context.Context, store.BusinessProfile) (dlc.Brand, error) {
	return dlc.Brand{}, ErrNotImplemented
}

func (Registrar) Brand(context.Context, string) (dlc.Brand, error) {
	return dlc.Brand{}, ErrNotImplemented
}

func (Registrar) RegisterCampaign(context.Context, string, store.UseCase, string) (dlc.Campaign, error) {
	return dlc.Campaign{}, ErrNotImplemented
}

func (Registrar) Campaign(context.Context, string) (dlc.Campaign, error) {
	return dlc.Campaign{}, ErrNotImplemented
}

func (Registrar) AssignNumber(context.Context, string, string) error { return ErrNotImplemented }

func (Registrar) VerifyHook(http.Header, []byte, time.Time) error { return ErrNotImplemented }

func (Registrar) HookSubject([]byte) (string, string) { return "", "" }
