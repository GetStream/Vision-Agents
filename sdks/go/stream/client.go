package stream

// Client is the acceleration backend held once: where the router is, who is calling it,
// and the credential that says so.
//
// It exists so that a program says that in one place rather than at every call site. What
// is routed through it reads the same whether the router is on localhost or behind the
// hosted proxy, because which one it is was decided here:
//
//	client, err := stream.NewClient(stream.Backend{})
//	transcriber, err := client.Router("healthcare").STT().Realtime(ctx, nil)
type Client struct {
	backend Backend
}

// NewClient resolves the backend and refuses one that cannot be reached or billed.
//
// The zero Backend reads the environment, which is what a program deployed next to a
// router wants. Resolving here rather than at the first call is the point of holding a
// client: a missing credential is reported where the backend was configured, not in the
// middle of a session that was about to stream audio.
func NewClient(backend Backend) (*Client, error) {
	resolved, err := backend.Resolve()
	if err != nil {
		return nil, err
	}
	return &Client{backend: resolved}, nil
}

// Router routes through a stored config, named by what it was stored under or by its id.
//
// A config is what makes "this is the healthcare setup" something said once rather than a
// set of options repeated at every call site. The empty name is a router told what to do
// per call instead.
func (c *Client) Router(config string) Router {
	return Router{Config: config, Backend: c.backend}
}

// Backend is where this client points, for the parts of the SDK that take one directly.
func (c *Client) Backend() Backend { return c.backend }
