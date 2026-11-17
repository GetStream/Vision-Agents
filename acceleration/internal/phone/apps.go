package phone

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// AppsOf makes a customer's phone lines in the Stream app it acts in, and finishes the lines
// already made in the app they were made in.
func AppsOf(clients *streamapp.Clients) Apps {
	return clientApps{clients: clients}
}

type clientApps struct {
	clients *streamapp.Clients
}

func (a clientApps) For(ctx context.Context, customer string) (*Stream, int64, error) {
	bound, err := a.clients.For(ctx, customer)
	if err != nil {
		return nil, 0, err
	}
	return NewStreamFromClient(bound.Client), bound.Identity.StreamApp, nil
}

func (a clientApps) ForApp(ctx context.Context, customer string, app int64) (*Stream, error) {
	bound, err := a.clients.ForApp(ctx, customer, app)
	if err != nil {
		return nil, err
	}
	return NewStreamFromClient(bound.Client), nil
}

func (a clientApps) ForAppRemoving(ctx context.Context, customer string, app int64) (*Stream, error) {
	bound, err := a.clients.ForAppReading(ctx, customer, app)
	if err != nil {
		return nil, err
	}
	return NewStreamFromClient(bound.Client), nil
}
