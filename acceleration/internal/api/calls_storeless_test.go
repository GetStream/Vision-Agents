package api

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// StorelessTokenSuite covers the token operations on a router run with no database, which
// holds no record of whose guests are whose.
type StorelessTokenSuite struct {
	suite.Suite
}

func TestStorelessTokenSuite(t *testing.T) {
	suite.Run(t, new(StorelessTokenSuite))
}

func (s *StorelessTokenSuite) TestANamedUserInASharedAppIsRefused() {
	// Nothing says the user is the customer's, so a shared app mints for nobody named. The
	// base URL answers nothing: a refusal is decided before Stream is asked.
	server := &Server{stream: streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: "key", Secret: "secret", App: 77, BaseURL: "http://127.0.0.1:1",
	}), streamapp.ClientsOptions{})}
	ctx := context.WithValue(context.Background(), customerContextKey{}, "acme")

	_, err := server.createChatToken(ctx, &createChatTokenRequest{Body: &ChatTokenRequest{
		AgentId: "support-agent", UserId: new("someone"),
	}})

	s.Equal(errUserInSharedApp, err)
}
