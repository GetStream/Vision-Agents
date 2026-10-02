package main

import (
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// userTokenEnvVar is a fixed token the voice edge has always preferred to minting its own.
// It is not a setting: it was only ever read raw, and only the deployment's app carries it.
const userTokenEnvVar = "STREAM_USER_TOKEN"

// newStreamClients builds what every Stream action the router takes resolves through. In
// deployment mode that is the deployment's own app, for every customer.
func newStreamClients(settings config.Config) *streamapp.Clients {
	deployment := streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey:    settings.Stream.APIKey,
		Secret:    settings.Stream.APISecret,
		UserToken: os.Getenv(userTokenEnvVar),
		BaseURL:   settings.Stream.BaseURL,
	})
	return streamapp.NewClients(deployment, streamapp.ClientsOptions{})
}
