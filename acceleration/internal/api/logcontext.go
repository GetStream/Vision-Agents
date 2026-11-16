package api

import (
	"context"
	"log/slog"
)

// The names a log line gives the app and the key a request was made with: the Stream
// platform's own field names.
const (
	logAppField    = "app_id"
	logAPIKeyField = "api_key"
)

// loggedAppContextKey holds the app and public key an authenticated request was made with,
// for the log handler. It is its own value rather than the customer's, which a scheduled
// delivery also sets on a context that is no request's.
type loggedAppContextKey struct{}

type loggedApp struct {
	appID, apiKey string
}

// WithLoggedApp is ctx marked with the app and public key to log beside every line written with it.
func WithLoggedApp(ctx context.Context, appID, apiKey string) context.Context {
	return context.WithValue(ctx, loggedAppContextKey{}, loggedApp{appID: appID, apiKey: apiKey})
}

// logHandler adds the app and the public key of an authenticated request to every record
// logged with its context (InfoContext and the rest). A request nobody authenticated, a hook
// among them, and a record logged without the request's context gain nothing.
type logHandler struct {
	slog.Handler
	// attached holds the fields a With already put on the logger, which Handle must not repeat.
	attached map[string]bool
}

// NewLogHandler wraps next so that records carry the app_id and api_key of the request
// whose context they are logged with.
func NewLogHandler(next slog.Handler) slog.Handler {
	return logHandler{Handler: next}
}

func (h logHandler) Handle(ctx context.Context, record slog.Record) error {
	app, ok := ctx.Value(loggedAppContextKey{}).(loggedApp)
	if !ok {
		return h.Handler.Handle(ctx, record)
	}
	named := map[string]bool{}
	record.Attrs(func(attr slog.Attr) bool {
		named[attr.Key] = true
		return true
	})
	if app.appID != "" && !named[logAppField] && !h.attached[logAppField] {
		record.AddAttrs(slog.String(logAppField, app.appID))
	}
	if app.apiKey != "" && !named[logAPIKeyField] && !h.attached[logAPIKeyField] {
		record.AddAttrs(slog.String(logAPIKeyField, app.apiKey))
	}
	return h.Handler.Handle(ctx, record)
}

func (h logHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	attached := map[string]bool{}
	for key := range h.attached {
		attached[key] = true
	}
	for _, attr := range attrs {
		attached[attr.Key] = true
	}
	return logHandler{Handler: h.Handler.WithAttrs(attrs), attached: attached}
}

func (h logHandler) WithGroup(name string) slog.Handler {
	// What Handle adds lands inside the group, so no field attached before it is repeated.
	return logHandler{Handler: h.Handler.WithGroup(name)}
}
