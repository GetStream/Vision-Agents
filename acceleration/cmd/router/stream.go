package main

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// newStreamClients builds what every Stream action the router takes resolves through. In
// deployment mode that is the deployment's own app, for every customer.
func newStreamClients(settings config.Config) *streamapp.Clients {
	deployment := streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey:    settings.Stream.APIKey,
		Secret:    settings.Stream.APISecret,
		UserToken: settings.Stream.UserToken,
		BaseURL:   settings.Stream.BaseURL,
	})
	return streamapp.NewClients(deployment, streamapp.ClientsOptions{})
}

// streamPins tells the store how this deployment's Stream app pins cross to another one.
func streamPins(clients *streamapp.Clients) store.StreamPins {
	return store.StreamPins{Deployment: clients.DeploymentApp, For: clients.Pin}
}
