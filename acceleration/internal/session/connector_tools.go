package session

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const defaultConnectorToolTimeout = 5 * time.Second

const (
	connectorAccountUnavailable    = "account_unavailable"
	connectorAccountNotAuthorized  = "account_not_authorized"
	connectorProviderMismatch      = "provider_mismatch"
	connectorNeedsReauthorization  = "needs_reauthorization"
	connectorCredentialUnavailable = "credential_unavailable"
	connectorToolUnavailable       = "tool_unavailable"
)

// attachConnectors resolves every configured grant against a tenant-owned connection and
// the verified caller before opening any remote MCP session.
func attachConnectors(
	ctx context.Context,
	spec Spec,
	db *store.Store,
	sealer *auth.Sealer,
	logger *slog.Logger,
	transport *http.Client,
) (*mcp.Runtime, []harness.Tool, []ConnectorUnavailable, error) {
	var unavailable []ConnectorUnavailable
	reportUnavailable := func(binding store.ConnectorBinding, reason string) {
		unavailable = append(unavailable, ConnectorUnavailable{
			Name: binding.Name, ConnectorID: binding.ConnectorID, Reason: reason,
		})
	}
	selected := make(map[string]string, len(spec.ConnectorSelections))
	for _, choice := range spec.ConnectorSelections {
		if choice.Name == "" || choice.ConnectionID == "" {
			return nil, nil, nil, fmt.Errorf("session: connector selections need a name and connection id")
		}
		if _, exists := selected[choice.Name]; exists {
			return nil, nil, nil, fmt.Errorf("session: connector %s was selected more than once", choice.Name)
		}
		selected[choice.Name] = choice.ConnectionID
	}
	if len(selected) > 0 && len(spec.ConnectorBindings) == 0 {
		return nil, nil, nil, fmt.Errorf("session: connector selections require an agent config binding")
	}
	if len(spec.ConnectorBindings) == 0 {
		return nil, nil, nil, nil
	}
	if spec.ConfigID == "" {
		return nil, nil, nil, fmt.Errorf("session: connector bindings require a stored agent config")
	}
	if db == nil {
		return nil, nil, nil, fmt.Errorf("session: connector storage is unavailable")
	}
	bindings := make(map[string]store.ConnectorBinding, len(spec.ConnectorBindings))
	connections := make([]mcp.Connection, 0, len(spec.ConnectorBindings))
	required := make(map[string]store.ConnectorBinding, len(spec.ConnectorBindings))
	authenticator := &mcp.OAuthClient{}
	for _, binding := range spec.ConnectorBindings {
		if binding.Name == "" {
			return nil, nil, nil, fmt.Errorf("session: connector binding has no alias")
		}
		if _, duplicate := bindings[binding.Name]; duplicate {
			return nil, nil, nil, fmt.Errorf("session: connector binding %s is duplicated", binding.Name)
		}
		bindings[binding.Name] = binding
		connectionID := ""
		switch binding.Connection.Type {
		case "fixed":
			connectionID = binding.Connection.ConnectionID
			if _, supplied := selected[binding.Name]; supplied {
				return nil, nil, nil, fmt.Errorf("session: connector %s has a fixed account and cannot be overridden", binding.Name)
			}
		case "session":
			connectionID = selected[binding.Name]
			delete(selected, binding.Name)
			if connectionID == "" {
				if binding.Required {
					return nil, nil, nil, fmt.Errorf("session: connector %s needs an explicit account selection", binding.Name)
				}
				continue
			}
		default:
			return nil, nil, nil, fmt.Errorf("session: connector %s has an invalid account selection type", binding.Name)
		}
		if connectionID == "" {
			return nil, nil, nil, fmt.Errorf("session: connector %s has no connection id", binding.Name)
		}
		connection, err := db.ConnectorConnection(ctx, spec.CustomerID, connectionID)
		if err != nil {
			if binding.Required {
				return nil, nil, nil, fmt.Errorf("session: required connector %s has no account connection", binding.Name)
			}
			reportUnavailable(binding, connectorAccountUnavailable)
			logger.Warn("optional connector account is unavailable", "connector", binding.Name)
			continue
		}
		if connection.ConnectorID != binding.ConnectorID {
			if binding.Required {
				return nil, nil, nil, fmt.Errorf("session: connector %s account belongs to another provider", binding.Name)
			}
			reportUnavailable(binding, connectorProviderMismatch)
			logger.Warn("optional connector account belongs to another provider", "connector", binding.Name)
			continue
		}
		if binding.Connection.Type == "fixed" {
			if connection.OwnerType != "app" {
				return nil, nil, nil, fmt.Errorf("session: fixed connector %s must use an app-owned connection", binding.Name)
			}
		} else if connection.OwnerType != "user" || spec.Caller.UserID == "" ||
			spec.CallerKind == auth.KindAnonymous || spec.CallerKind == auth.KindGuest ||
			connection.OwnerID != spec.Caller.UserID {
			return nil, nil, nil, fmt.Errorf("session: connector %s account does not belong to the verified caller", binding.Name)
		}
		if connection.Status != store.ConnectorConnected {
			if binding.Required {
				return nil, nil, nil, fmt.Errorf("session: required connector %s needs reauthorization", binding.Name)
			}
			if connection.Status == store.ConnectorNeedsReauth {
				reportUnavailable(binding, connectorNeedsReauthorization)
			} else {
				reportUnavailable(binding, connectorAccountUnavailable)
			}
			logger.Warn("optional connector needs reauthorization", "connector", binding.Name, "status", connection.Status)
			continue
		}
		allowed := make([]string, 0, len(binding.Tools))
		expectedToolDigests := make(map[string]string, len(binding.Tools))
		for _, grant := range binding.Tools {
			allowed = append(allowed, grant.Name)
			expectedToolDigests[grant.Name] = grant.SchemaDigest
		}
		timeout := defaultConnectorToolTimeout
		if binding.TimeoutMs > 0 {
			timeout = time.Duration(binding.TimeoutMs) * time.Millisecond
		}
		if _, err := connectors.ResolveCredentials(ctx, db, sealer, spec.CustomerID, connection.ID, authenticator); err != nil {
			if binding.Required {
				return nil, nil, nil, fmt.Errorf("session: required connector %s needs reauthorization", binding.Name)
			}
			if errors.Is(err, connectors.ErrReauthorizationRequired) {
				reportUnavailable(binding, connectorNeedsReauthorization)
			} else {
				reportUnavailable(binding, connectorCredentialUnavailable)
			}
			logger.Warn("optional connector could not resolve its authorization", "connector", binding.Name)
			continue
		}
		boundBinding := binding
		boundConnectionID := connection.ID
		boundEndpoint := connection.Endpoint
		connections = append(connections, mcp.Connection{
			ConnectorID:         binding.Name,
			ToolPrefix:          binding.Name,
			Endpoint:            connection.Endpoint,
			AllowedTools:        allowed,
			ExpectedToolDigests: expectedToolDigests,
			Authorize: func(callCtx context.Context, request *http.Request) error {
				return connectors.AuthorizeRequest(callCtx, db, sealer, spec.CustomerID, boundConnectionID, authenticator, request)
			},
			AuthorizeTool: func(callCtx context.Context, toolName, schemaDigest string) error {
				return authorizeConnectorTool(
					callCtx, db, spec, boundBinding, boundConnectionID, boundEndpoint, toolName, schemaDigest,
				)
			},
			Timeout: timeout,
		})
		if binding.Required {
			required[binding.Name] = binding
		}
	}
	if len(selected) > 0 {
		for name := range selected {
			return nil, nil, nil, fmt.Errorf("session: connector %s is not a session-selected config binding", name)
		}
	}
	if len(connections) == 0 {
		return nil, nil, unavailable, nil
	}
	runtime, tools, failures := mcp.Open(ctx, connections, transport)
	for _, failure := range failures {
		alias, _, _ := strings.Cut(failure.Error(), ":")
		if binding, isRequired := required[alias]; isRequired {
			if runtime != nil {
				runtime.Close()
			}
			return nil, nil, nil, fmt.Errorf("session: required connector %s could not be opened", binding.Name)
		}
		if binding, exists := bindings[alias]; exists {
			reason := connectorAccountUnavailable
			if strings.Contains(failure.Error(), "tool") {
				reason = connectorToolUnavailable
			}
			reportUnavailable(binding, reason)
		}
		logger.Warn("optional connector could not be opened", "error", failure)
	}
	if runtime == nil {
		return nil, nil, unavailable, nil
	}
	present := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		present[tool.Name] = struct{}{}
	}
	for alias, binding := range required {
		for _, grant := range binding.Tools {
			if _, exists := present[mcp.Prefix(alias, grant.Name)]; !exists {
				runtime.Close()
				return nil, nil, nil, fmt.Errorf("session: required connector %s does not offer granted tool %s", alias, grant.Name)
			}
		}
	}
	return runtime, tools, unavailable, nil
}

func authorizeConnectorTool(
	ctx context.Context,
	db *store.Store,
	spec Spec,
	initial store.ConnectorBinding,
	connectionID, endpoint, toolName, schemaDigest string,
) error {
	config, err := db.AgentConfig(ctx, spec.CustomerID, spec.ConfigID)
	if err != nil {
		return fmt.Errorf("session: connector %s agent binding is no longer available", initial.Name)
	}
	var current *store.ConnectorBinding
	for index := range config.Connectors {
		if config.Connectors[index].Name == initial.Name {
			current = &config.Connectors[index]
			break
		}
	}
	if current == nil || current.ConnectorID != initial.ConnectorID || current.Connection.Type != initial.Connection.Type {
		return fmt.Errorf("session: connector %s agent binding is no longer authorized", initial.Name)
	}

	connection, err := db.ConnectorConnection(ctx, spec.CustomerID, connectionID)
	if err != nil {
		return fmt.Errorf("session: connector %s account is no longer authorized", initial.Name)
	}
	if connection.ConnectorID != current.ConnectorID {
		return fmt.Errorf("session: connector %s account provider changed while the session was open", initial.Name)
	}
	if connection.Endpoint != endpoint {
		return fmt.Errorf("session: connector %s account endpoint changed while the session was open", initial.Name)
	}
	switch current.Connection.Type {
	case "fixed":
		if current.Connection.ConnectionID != connectionID || connection.OwnerType != "app" {
			return fmt.Errorf("session: connector %s fixed account is no longer authorized", initial.Name)
		}
	case "session":
		selected := false
		for _, choice := range spec.ConnectorSelections {
			if choice.Name == initial.Name && choice.ConnectionID == connectionID {
				selected = true
				break
			}
		}
		if !selected || connection.OwnerType != "user" || spec.Caller.UserID == "" ||
			spec.CallerKind == auth.KindAnonymous || spec.CallerKind == auth.KindGuest ||
			connection.OwnerID != spec.Caller.UserID {
			return fmt.Errorf("session: connector %s account is no longer authorized for the verified caller", initial.Name)
		}
	default:
		return fmt.Errorf("session: connector %s account selection is no longer valid", initial.Name)
	}
	for _, grant := range current.Tools {
		if grant.Name == toolName && grant.SchemaDigest == schemaDigest {
			return nil
		}
	}
	return fmt.Errorf("session: connector %s tool %s is no longer granted", initial.Name, toolName)
}
