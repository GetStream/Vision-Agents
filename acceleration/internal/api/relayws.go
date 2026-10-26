package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/relay"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// attachWait is how long a socket waits for whichever node holds the session to answer.
// Nothing answering means no node has it, which is a session that does not exist.
const attachWait = 2 * time.Second

// remoteFrames is how many frames a relayed socket may fall behind by. It matches the
// buffer a local watcher gets, for the same reason: a control channel whose reader has
// stopped reading costs that reader some detail and nobody on the call anything. The other
// half is room for what a resumable watcher is replayed when it attaches.
const remoteFrames = 512

// remoteCommands is how many commands one relayed socket may have waiting. A watcher
// sends them one at a time and waits for what they do, so a queue this deep is already
// more than a client can get ahead by.
const remoteCommands = 16

// relayed is what a node needs to take part in a session another node is running: the
// sockets it holds onto conversations elsewhere, and the watchers it holds on behalf of
// sockets elsewhere.
//
// Both halves are here rather than in the relay package because what is being carried is
// this package's own: the frames are frameOf's and the commands are applyCommand's.
type relayed struct {
	bus *relay.Bus
	// filter answers whether an event crossing the bus can be for a socket here. Every
	// node is sent every session's events, so it is asked before the registry's lock is.
	filter *relay.Filter

	mu sync.Mutex
	// sockets are the relayed sockets this node is writing to, by watcher id.
	sockets map[string]*remoteSocket
	// proxies are the watchers this node attached for a socket elsewhere, by the same id.
	proxies map[string]*proxyWatcher
}

// remoteSocket is one socket here onto a session somewhere else.
type remoteSocket struct {
	// owner is the session's owner key, which is only known once the holder has answered
	// and is what the filter entry has to be given back under. Guarded by relayed.mu.
	owner string
	// attached carries the holder's answer, and is what the handler waits on before it
	// upgrades: a socket opened before the answer would be one nothing is watching.
	attached chan string
	frames   chan json.RawMessage
	closed   chan struct{}
	once     sync.Once
	// halted is set before closed when the holder's router stopped the session.
	halted atomic.Bool
}

func (r *remoteSocket) close() { r.once.Do(func() { close(r.closed) }) }

// proxyWatcher is one watcher attached here for a socket on another node.
type proxyWatcher struct {
	detach func()
	// owner is who the socket's caller was when it attached, and is what its commands are
	// carried out as. It is kept rather than read off each command so that a node cannot
	// change who it claims to be halfway through a conversation. seat is the same caller
	// with the tools it said it runs.
	owner session.Owner
	seat  session.Seat
	// commands are what that caller sent, carried out one at a time by a goroutine of
	// this watcher's own. They cannot be run where they arrive: every node's requests
	// come in on one subscription, and answering a question through the model takes as
	// long as the model does.
	commands chan json.RawMessage
	// seen is when the socket's node last said it was still open. A node that stopped
	// saying so has gone, and the watcher it left behind would otherwise hold a
	// persistent conversation open and go on being offered the model's tool calls.
	seen time.Time
}

// newRelayed starts both subscriptions and returns once they are live, so a session
// opened immediately afterwards can already be reached from another node.
func (s *Server) newRelayed(ctx context.Context, bus *relay.Bus) (*relayed, error) {
	state := &relayed{
		bus:     bus,
		filter:  relay.NewFilter(),
		sockets: map[string]*remoteSocket{},
		proxies: map[string]*proxyWatcher{},
	}
	if err := bus.Subscribe(ctx, bus.Events(), state.deliver); err != nil {
		return nil, err
	}
	if err := bus.Subscribe(ctx, bus.Commands(), func(payload []byte) {
		s.serveRelayRequest(ctx, state, payload)
	}); err != nil {
		return nil, err
	}
	go s.expireProxies(ctx, state)
	return state, nil
}

// watchRemoteSession serves a watcher whose session is running on another node, reporting
// whether it answered the request. It answers nothing when no node let this caller watch, so
// the caller can be answered as for a session nobody is running.
//
// Who may watch is decided by the node holding the session rather than here: it has the
// session's own spec, where this node has at best a row written behind it. A node is not
// trusted to have checked, so the caller is named on the bus and checked there.
func (s *Server) watchRemoteSession(w http.ResponseWriter, r *http.Request, id string, owner session.Owner) bool {
	state := s.relayed
	seat := seatOf(r, owner)
	asked := resumingOf(r)
	watcher := uuid.NewString()
	socket := &remoteSocket{
		attached: make(chan string, 1),
		frames:   make(chan json.RawMessage, remoteFrames),
		closed:   make(chan struct{}),
	}

	state.mu.Lock()
	state.sockets[watcher] = socket
	state.mu.Unlock()
	attached := false
	defer func() {
		if !attached {
			state.detach(context.WithoutCancel(r.Context()), watcher, id)
		}
	}()

	if err := state.bus.Publish(r.Context(), state.bus.Commands(), relay.Request{
		Type:    relay.Attach,
		Node:    state.bus.Node(),
		Session: id,
		Watcher: watcher,
		Owner: relay.Owner{
			CustomerID: owner.CustomerID,
			UserID:     owner.UserID,
			Kind:       string(owner.Kind),
		},
		ReplayPendingTools: seat.ReplayPendingVoiceTools,
		Tools:              seat.Tools,
		DeclaresTools:      seat.Declares,
		Resumable:          asked.resumable,
		Cursor:             asked.cursor,
		RequestID:          asked.requestID,
	}); err != nil {
		s.logger.Error("could not ask for a session held elsewhere", "session", id, "error", err)
		writeError(w, unavailable("the session could not be reached"))
		return true
	}

	select {
	case <-socket.attached:
		attached = true
	case <-time.After(attachWait):
		// No node has it, or the node that has it will not let this caller watch. Both
		// are answered the way a session that never existed is answered: a refusal would
		// confirm the id is real.
		return false
	case <-r.Context().Done():
		return true
	}

	connection, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Debug("could not upgrade a relayed session socket", "error", err)
		state.detach(context.WithoutCancel(r.Context()), watcher, id)
		return true
	}
	defer connection.Close()
	defer state.detach(context.WithoutCancel(r.Context()), watcher, id)
	connection.SetReadLimit(maxSocketMessage)

	gone := make(chan error, 1)
	go func() {
		gone <- s.readRelayedCommands(connection, state, watcher, id)
	}()
	s.writeRelayedEvents(connection, state, socket, watcher, id, watching(r), gone, endingOf(asked, socket.halted.Load))
	return true
}

// writeRelayedEvents writes what arrives on the bus out to the caller, and keeps both the
// socket and the watcher behind it alive.
//
// The ping and the refresh share a tick because they say the same thing in each
// direction: this socket is still being read, and the watcher standing for it elsewhere
// is still wanted.
func (s *Server) writeRelayedEvents(
	connection *websocket.Conn,
	state *relayed,
	socket *remoteSocket,
	watcher, id string,
	asked wanted,
	gone <-chan error,
	ended ending,
) {
	ping := time.NewTicker(pingEvery)
	defer ping.Stop()

	for {
		select {
		case stopped := <-gone:
			if errors.Is(stopped, errUnreadableFrame) {
				refuseUnreadable(connection)
			}
			return

		case <-socket.closed:
			code, reason := ended()
			closeBeforeReader(connection, code, reason, gone)
			return

		case payload := <-socket.frames:
			if !asked.takesFrame(frameType(payload)) {
				continue
			}
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteMessage(websocket.TextMessage, payload); err != nil {
				s.logger.Debug("relayed session socket write failed", "error", err)
				return
			}

		case <-ping.C:
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
			if err := state.bus.Publish(context.Background(), state.bus.Commands(), relay.Request{
				Type: relay.Refresh, Node: state.bus.Node(), Session: id, Watcher: watcher,
			}); err != nil {
				s.logger.Debug("could not refresh a relayed watcher", "session", id, "error", err)
			}
		}
	}
}

// readRelayedCommands carries what the caller sends to the node holding the session.
//
// The frame is passed on as it arrived rather than decoded and encoded again, so the
// holder reads exactly what the client wrote and there is one place -- applyCommand --
// that decides what a command means. It is decoded here all the same, so a frame that is not
// a command ends this socket as it ends a local one; the holder would only report it.
//
// It returns errUnreadableFrame for a frame that is not a command, and the read's own error
// when the connection ended.
func (s *Server) readRelayedCommands(connection *websocket.Conn, state *relayed, watcher, id string) error {
	connection.SetReadDeadline(time.Now().Add(pongWait))
	connection.SetPongHandler(func(string) error {
		return connection.SetReadDeadline(time.Now().Add(pongWait))
	})

	for {
		_, payload, err := connection.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.logger.Debug("relayed session socket read failed", "error", err)
			}
			return stack.Wrap(err)
		}
		connection.SetReadDeadline(time.Now().Add(pongWait))
		var command watcherCommand
		if json.Unmarshal(payload, &command) != nil {
			return errUnreadableFrame
		}

		if err := state.bus.Publish(context.Background(), state.bus.Commands(), relay.Request{
			Type:    relay.Command,
			Node:    state.bus.Node(),
			Session: id,
			Watcher: watcher,
			Payload: payload,
		}); err != nil {
			s.logger.Error("could not carry a command to the node holding the session",
				"session", id, "error", err)
		}
	}
}

// serveRelayRequest answers another node's request about a session this one is holding.
//
// A request about a session nothing here is running is left unanswered rather than
// refused: every node sees every request, so most of them are not this node's to answer
// and a refusal from each would be noise the asker has to tell apart from a real no.
func (s *Server) serveRelayRequest(ctx context.Context, state *relayed, payload []byte) {
	var request relay.Request
	if err := json.Unmarshal(payload, &request); err != nil {
		s.logger.Debug("unreadable relay request", "error", err)
		return
	}
	if request.Node == state.bus.Node() || s.sessions == nil {
		return
	}

	switch request.Type {
	case relay.Attach:
		s.attachProxy(ctx, state, request)
	case relay.Refresh:
		state.refresh(request.Watcher)
	case relay.Detach:
		state.dropProxy(request.Watcher)
	case relay.Command:
		state.queueCommand(request.Watcher, request.Payload)
	}
}

// attachProxy watches a session here on behalf of a socket elsewhere.
//
// The watcher is a real one, which is the point: the session cannot tell a socket on
// another node from one on this one, so a tool call still finds somebody to answer it and
// a persistent conversation still ends when the last watcher goes away.
func (s *Server) attachProxy(ctx context.Context, state *relayed, request relay.Request) {
	owner := session.Owner{
		CustomerID: request.Owner.CustomerID,
		UserID:     request.Owner.UserID,
		Kind:       auth.Kind(request.Owner.Kind),
	}
	found, ok := s.sessions.Get(request.Session, owner)
	if !ok {
		return
	}
	// The same question canReadSession asks of a local watcher: a saved conversation is
	// the caller's own or it is nobody's.
	spec := found.Spec()
	if spec.PersistConversation && spec.Caller.UserID != owner.UserID {
		return
	}

	held := session.OwnerOf(spec)
	key := relay.Owner{CustomerID: held.CustomerID, UserID: held.UserID}.Key()

	seat := session.Seat{
		Owner: owner, Tools: request.Tools, Declares: request.DeclaresTools,
		ReplayPendingVoiceTools: request.ReplayPendingTools,
	}
	var events <-chan session.Event
	var detach func()
	if request.Resumable {
		var err error
		if events, detach, err = found.WatchFrom(seat, request.Cursor, request.RequestID); err != nil {
			return
		}
	} else {
		events, detach = found.WatchAs(seat)
	}

	proxy := &proxyWatcher{
		detach:   detach,
		owner:    owner,
		seat:     seat,
		commands: make(chan json.RawMessage, remoteCommands),
		seen:     time.Now(),
	}
	if !state.holdProxy(request.Watcher, proxy) {
		// The socket asked twice, which a retried attach is. The watcher already
		// attached for it stands, and this one is given back rather than left running.
		detach()
		return
	}

	// Answered before any frame is published, so the asking node holds the key its
	// filter is keyed on before there is anything to filter.
	if err := state.bus.Publish(ctx, state.bus.Events(), relay.Response{
		Type:    relay.Attached,
		Node:    state.bus.Node(),
		Owner:   key,
		Watcher: request.Watcher,
	}); err != nil {
		s.logger.Error("could not answer an attach", "session", request.Session, "error", err)
		state.dropProxy(request.Watcher)
		return
	}

	go s.runRelayedCommands(found, proxy)
	go s.publishSessionEvents(ctx, state, events, request.Watcher, request.Session, key, found.Halted)
}

// publishSessionEvents puts one session's events on the bus for the socket watching it
// from another node.
func (s *Server) publishSessionEvents(
	ctx context.Context,
	state *relayed,
	events <-chan session.Event,
	watcher, id, key string,
	halted func() bool,
) {
	// The events channel closing is either the session ending or the watcher being
	// detached, and both end the watcher: dropping it here is what closes its command
	// queue and takes it out of the registry.
	defer state.dropProxy(watcher)

	for event := range events {
		rendered, ok := frameOf(event)
		if !ok {
			continue
		}
		payload, err := json.Marshal(rendered)
		if err != nil {
			s.logger.Debug("could not render a relayed event", "session", id, "error", err)
			continue
		}
		if err := state.bus.Publish(ctx, state.bus.Events(), relay.Response{
			Type:    relay.Frame,
			Node:    state.bus.Node(),
			Owner:   key,
			Watcher: watcher,
			Payload: payload,
		}); err != nil {
			s.logger.Debug("could not relay a session event", "session", id, "error", err)
		}
	}

	if err := state.bus.Publish(ctx, state.bus.Events(), relay.Response{
		Type:    relay.Closed,
		Node:    state.bus.Node(),
		Owner:   key,
		Watcher: watcher,
		Halted:  halted(),
	}); err != nil {
		s.logger.Debug("could not relay the end of a session", "session", id, "error", err)
	}
}

// runRelayedCommands carries out what the socket on the other node sends, one at a time,
// through the same switch a local socket's commands go through.
func (s *Server) runRelayedCommands(found *session.Session, proxy *proxyWatcher) {
	for payload := range proxy.commands {
		var command watcherCommand
		if err := json.Unmarshal(payload, &command); err != nil {
			found.Report(err, "command")
			continue
		}
		// The other node holds the socket's address, so only its user is counted here.
		sender := routing.Caller{UserID: proxy.owner.UserID, ServerSide: proxy.owner.Kind == auth.KindServer,
			Verified: proxy.owner.UserID != "" && proxy.owner.Kind.Verified()}
		s.applyCommand(found, proxy.seat, sender, command)
	}
}

// expireProxies gives back the watchers whose sockets have gone quiet, which is what a
// node that crashed leaves behind.
func (s *Server) expireProxies(ctx context.Context, state *relayed) {
	ticker := time.NewTicker(pingEvery)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			for _, watcher := range state.expired() {
				s.logger.Debug("dropping a relayed watcher whose node went quiet",
					"watcher", watcher)
				state.dropProxy(watcher)
			}
		case <-ctx.Done():
			return
		}
	}
}

// deliver hands one message off the bus to the socket it is for, and is the hot path:
// every node is sent every session's events, so the first question asked is the cheap one.
func (r *relayed) deliver(payload []byte) {
	var response relay.Response
	if err := json.Unmarshal(payload, &response); err != nil {
		return
	}
	if response.Node == r.bus.Node() {
		return
	}
	// An answer to an attach is what tells this node the key to filter on, so it cannot
	// itself be filtered on it. There is one per socket, where there are many frames a
	// second.
	if response.Type == relay.Attached {
		r.attach(response.Watcher, response.Owner)
		return
	}
	if !r.filter.Has(response.Owner) {
		return
	}

	r.mu.Lock()
	socket := r.sockets[response.Watcher]
	r.mu.Unlock()
	if socket == nil {
		return
	}

	if response.Type == relay.Closed {
		if response.Halted {
			socket.halted.Store(true)
		}
		socket.close()
		return
	}
	select {
	case socket.frames <- response.Payload:
	default:
	}
}

// attach records that the holder answered, putting the key in the filter before the
// handler is let go, so that no frame can arrive before the filter would pass it.
func (r *relayed) attach(watcher, owner string) {
	r.mu.Lock()
	socket := r.sockets[watcher]
	if socket != nil && socket.owner == "" {
		socket.owner = owner
		r.filter.Add(owner)
	}
	r.mu.Unlock()
	if socket == nil {
		return
	}

	select {
	case socket.attached <- owner:
	default:
	}
}

// detach forgets a socket, gives back its filter entry, and tells the node holding the
// session that the watcher standing for it is no longer wanted.
func (r *relayed) detach(ctx context.Context, watcher, id string) {
	r.mu.Lock()
	socket, held := r.sockets[watcher]
	if held {
		delete(r.sockets, watcher)
		if socket.owner != "" {
			r.filter.Remove(socket.owner)
		}
	}
	r.mu.Unlock()
	if held {
		socket.close()
	}

	_ = r.bus.Publish(ctx, r.bus.Commands(), relay.Request{
		Type: relay.Detach, Node: r.bus.Node(), Session: id, Watcher: watcher,
	})
}

// holdProxy records a watcher attached for a socket elsewhere, reporting false when one
// is already held under that id.
func (r *relayed) holdProxy(watcher string, proxy *proxyWatcher) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, already := r.proxies[watcher]; already {
		return false
	}
	r.proxies[watcher] = proxy
	return true
}

// dropProxy detaches a watcher held for a socket elsewhere. Detaching closes the session's
// events channel, which is what ends the goroutine publishing from it, and closing the
// queue is what ends the one carrying out its commands.
//
// Taking it out of the map under the lock is what makes this safe to call more than once:
// a second caller finds nothing and does nothing.
func (r *relayed) dropProxy(watcher string) {
	r.mu.Lock()
	held, found := r.proxies[watcher]
	delete(r.proxies, watcher)
	r.mu.Unlock()
	if !found {
		return
	}

	held.detach()
	close(held.commands)
}

// queueCommand hands one command to the watcher it names, for that watcher's own
// goroutine to carry out. A command naming a watcher nothing here holds belongs to
// another node's proxy.
func (r *relayed) queueCommand(watcher string, payload json.RawMessage) {
	r.mu.Lock()
	defer r.mu.Unlock()

	held, found := r.proxies[watcher]
	if !found {
		return
	}
	// Under the lock, so the queue cannot be closed between finding it and sending.
	select {
	case held.commands <- payload:
	default:
	}
}

func (r *relayed) refresh(watcher string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if held, found := r.proxies[watcher]; found {
		held.seen = time.Now()
	}
}

// expired names the watchers whose sockets have not been heard from for longer than a
// socket is given to answer a ping.
func (r *relayed) expired() []string {
	r.mu.Lock()
	defer r.mu.Unlock()

	var stale []string
	for watcher, held := range r.proxies {
		if time.Since(held.seen) > pongWait {
			stale = append(stale, watcher)
		}
	}
	return stale
}

// frameType reads the type off a rendered event, which is all a relayed socket needs in
// order to decide whether its caller asked for it.
func frameType(payload json.RawMessage) string {
	var named struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(payload, &named); err != nil {
		return ""
	}
	return named.Type
}
