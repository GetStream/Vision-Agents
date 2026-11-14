package siptrunk

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strconv"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// holdFor is how long a node's claim on a call lasts unrenewed. A node that dies leaves its
// calls unheld for at most this long; the audio flows between the carrier and Stream without
// it, so only a re-INVITE or BYE in that time is lost.
const holdFor = 30 * time.Second

// renewEvery renews well inside holdFor, so a Redis that drops a write has more chances
// before another node takes a call this one still holds.
const renewEvery = 10 * time.Second

// keyPrefix holds every key in one hash slot: the scripts touch a call's keys and the set of
// calls together, which a Redis cluster refuses across slots.
const keyPrefix = "{siptrunk}"

// recordFor outlives the longest call, so a record a node never deleted goes by itself.
const recordFor = phone.MaxCallLength + time.Hour

// scanBatch caps the calls one scan takes, so a node that comes up after many died does not
// spend a whole renewal taking them.
const scanBatch = 100

// handoffPoll is how often Close looks whether another node took a call it offered.
const handoffPoll = 100 * time.Millisecond

// redisTimeout bounds a write about one call.
const redisTimeout = 5 * time.Second

// Resumer opens what continuing a call another node placed needs: both sides' credentials,
// read fresh, and how to report the call's end. *phone.Service is one.
type Resumer interface {
	ResumeTrunkCall(ctx context.Context, ref phone.TrunkCallRef) (phone.ResumedTrunkCall, error)
}

// callRecord is a connected call as Redis holds it, for another node to continue. It holds no
// password: Ref says where the node taking the call over reads them.
type callRecord struct {
	ID    string
	Ref   phone.TrunkCallRef
	State sipbridge.CallState
}

// keys names the Redis keys: a call's record, its owner, the set of calls scored by when
// each one's claim lapses, and the channel a stopping node offers its calls on.
type keys string

func (k keys) record(id string) string { return string(k) + ":call:" + id }
func (k keys) owner(id string) string  { return string(k) + ":owner:" + id }
func (k keys) calls() string           { return string(k) + ":calls" }
func (k keys) free() string            { return string(k) + ":free" }

// claimScript records a call and claims it for ARGV[1], unless another node holds it.
// KEYS: record, owner, calls. ARGV: node, record, recordFor ms, holdFor ms, lapse ms, id.
var claimScript = rueidis.NewLuaScript(`
local owner = redis.call('GET', KEYS[2])
if owner and owner ~= ARGV[1] then
  return 0
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
redis.call('SET', KEYS[2], ARGV[1], 'PX', ARGV[4])
redis.call('ZADD', KEYS[3], ARGV[5], ARGV[6])
return 1
`)

// writeScript rewrites a call's record, only for its owner: a node that gave the call up
// must not overwrite the record of the node that took it.
// KEYS: record, owner. ARGV: node, record, recordFor ms.
var writeScript = rueidis.NewLuaScript(`
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return 0
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
return 1
`)

// advanceScript rewrites a call's record for its owner, as writeScript does, but only while the
// record is there: says -1 when the call ended meanwhile, so a node taking it over does not
// bring it back.
// KEYS: record, owner. ARGV: node, record, recordFor ms.
var advanceScript = rueidis.NewLuaScript(`
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return 0
end
if not redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3], 'XX') then
  return -1
end
return 1
`)

// renewScript writes a call's record again and extends ARGV[1]'s claim. It says 0 when another
// node holds the claim, and 2, changing nothing, when no node does because the call was given up
// to another node. A claim gone while ARGV[7] says this node's lease still runs is Redis losing
// it, since no scan can have taken the call yet, so it is won again and says 3: letting the call
// go would hang up a call no node could take over without its record. Past the lease another
// node may have taken the call and ended it, so it says 0.
// KEYS: record, owner, calls. ARGV: node, record, recordFor ms, holdFor ms, lapse ms, id, leased.
var renewScript = rueidis.NewLuaScript(`
local owner = redis.call('GET', KEYS[2])
if owner == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
  redis.call('PEXPIRE', KEYS[2], ARGV[4])
  redis.call('ZADD', KEYS[3], ARGV[5], ARGV[6])
  return 1
end
if owner then
  return 0
end
if redis.call('ZSCORE', KEYS[3], ARGV[6]) == '0' then
  return 2
end
if ARGV[7] ~= '1' then
  return 0
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
redis.call('SET', KEYS[2], ARGV[1], 'PX', ARGV[4])
redis.call('ZADD', KEYS[3], ARGV[5], ARGV[6])
return 3
`)

// giveUpScript writes the call's record one last time and gives up ARGV[1]'s claim, marking
// it lapsed now so the next scan takes it.
// KEYS: record, owner, calls. ARGV: node, record, recordFor ms, id.
var giveUpScript = rueidis.NewLuaScript(`
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return 0
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
redis.call('DEL', KEYS[2])
redis.call('ZADD', KEYS[3], 0, ARGV[4])
return 1
`)

// forgetScript deletes an ended call, and its claim only when ARGV[1] holds it: a node
// still taking it over keeps its own.
// KEYS: record, owner, calls. ARGV: node, id.
var forgetScript = rueidis.NewLuaScript(`
redis.call('DEL', KEYS[1])
if redis.call('GET', KEYS[2]) == ARGV[1] then
  redis.call('DEL', KEYS[2])
end
redis.call('ZREM', KEYS[3], ARGV[2])
return 1
`)

// SetResumer gives the provider what opens a call's credentials to take it over. Call it
// before Start.
func (p *Provider) SetResumer(r Resumer) { p.resumer = r }

// Start renews this node's claims and, with a Resumer, takes over the calls other nodes gave
// up or stopped renewing, until ctx ends or Close. Without Redis it does nothing.
func (p *Provider) Start(ctx context.Context) {
	if p.redis == nil {
		return
	}
	ctx, cancel := context.WithCancel(ctx)
	p.mu.Lock()
	p.stopLoops = cancel
	p.mu.Unlock()
	p.loops.Go(func() { p.renewLoop(ctx) })
	if p.resumer == nil {
		return
	}
	offered := make(chan struct{}, 1)
	p.loops.Go(func() { p.listen(ctx, offered) })
	p.loops.Go(func() { p.scanLoop(ctx, offered) })
}

func (p *Provider) renewLoop(ctx context.Context) {
	ticker := time.NewTicker(p.renewEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.renew(ctx)
		}
	}
}

// renew writes the record of every call this node holds again, so it keeps CSeqs that moved
// with no change of SDP, extends the claim on each, and lets go of each one another node now
// holds.
func (p *Provider) renew(ctx context.Context) {
	p.mu.Lock()
	held := make(map[string]*heldCall, len(p.calls))
	for id, h := range p.calls {
		if h.movable {
			held[id] = h
		}
	}
	p.mu.Unlock()

	for id, h := range held {
		p.renewOne(ctx, id, h)
	}
}

func (p *Provider) renewOne(ctx context.Context, id string, h *heldCall) {
	// Held across the script, so a call that ends meanwhile is not written back after hold
	// deleted its keys.
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.ended {
		return
	}
	record, err := json.Marshal(callRecord{ID: id, Ref: *h.ref, State: h.call.State()})
	if err != nil {
		p.logger.Warn("could not record a call through a customer's sip trunk to renew its claim",
			"vendor_call_id", id, "error", err)
		return
	}
	// Bounded: hold waits on the lock to report the call's end.
	scriptCtx, cancel := context.WithTimeout(ctx, redisTimeout)
	defer cancel()
	now := time.Now()
	leased := "0"
	if now.Before(h.lapse) {
		leased = "1"
	}
	lapse := now.Add(p.holdFor)
	kept, err := renewScript.Exec(scriptCtx, p.redis, []string{p.keys.record(id), p.keys.owner(id), p.keys.calls()},
		[]string{p.node, string(record), ms(recordFor), ms(p.holdFor), strconv.FormatInt(lapse.UnixMilli(), 10), id, leased}).AsInt64()
	if err != nil {
		if ctx.Err() == nil {
			p.logger.Warn("could not renew the claim on a call through a customer's sip trunk",
				"vendor_call_id", id, "error", err)
		}
		return
	}
	switch kept {
	case 0:
		// Another node took it, or will: two nodes relaying one call would answer each
		// other's re-INVITEs out of order.
		h.call.Release()
	case 1:
		h.lapse = lapse
	case 3:
		h.lapse = lapse
		p.logger.Warn("claimed a call again after its claim went missing", "vendor_call_id", id)
	}
}

// listen tells offered each time another node gives up a call.
func (p *Provider) listen(ctx context.Context, offered chan<- struct{}) {
	for {
		err := p.redis.Receive(ctx, p.redis.B().Subscribe().Channel(p.keys.free()).Build(), func(rueidis.PubSubMessage) {
			select {
			case offered <- struct{}{}:
			default:
			}
		})
		if ctx.Err() != nil {
			return
		}
		p.logger.Warn("lost the channel other nodes offer their calls on; finding them by scanning only", "error", err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(p.renewEvery):
		}
	}
}

func (p *Provider) scanLoop(ctx context.Context, offered <-chan struct{}) {
	ticker := time.NewTicker(p.renewEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		case <-offered:
		}
		p.scan(ctx)
	}
}

// scan takes over the calls whose claims lapsed.
func (p *Provider) scan(ctx context.Context) {
	now := strconv.FormatInt(time.Now().UnixMilli(), 10)
	ids, err := p.redis.Do(ctx, p.redis.B().Zrangebyscore().Key(p.keys.calls()).Min("-inf").Max(now).
		Limit(0, scanBatch).Build()).AsStrSlice()
	if err != nil {
		if ctx.Err() == nil {
			p.logger.Warn("could not look for calls through customers' sip trunks no node holds", "error", err)
		}
		return
	}
	for _, id := range ids {
		if ctx.Err() != nil {
			return
		}
		p.take(id)
	}
}

// take continues a call no node holds, if this node wins it.
func (p *Provider) take(id string) {
	p.mu.Lock()
	_, mine := p.calls[id]
	p.mu.Unlock()
	if mine {
		return
	}
	log := p.logger.With("vendor", phone.SIPTrunkVendor, "vendor_call_id", id)
	// Not the loop's context: stopping this node mid-resume would hang the call up. It ends
	// before the claim won here lapses, so no other node resumes the call meanwhile.
	ctx, cancel := context.WithTimeout(context.Background(), p.holdFor-p.renewEvery)
	defer cancel()

	err := p.redis.Do(ctx, p.redis.B().Set().Key(p.keys.owner(id)).Value(p.node).Nx().
		PxMilliseconds(p.holdFor.Milliseconds()).Build()).Error()
	if rueidis.IsRedisNil(err) {
		return
	}
	if err != nil {
		log.Warn("could not claim a call through a customer's sip trunk no node holds", "error", err)
		return
	}

	raw, err := p.redis.Do(ctx, p.redis.B().Get().Key(p.keys.record(id)).Build()).ToString()
	if rueidis.IsRedisNil(err) {
		p.forget(id, log)
		return
	}
	if err != nil {
		log.Warn("could not read a call through a customer's sip trunk to take it over", "error", err)
		return
	}
	var rec callRecord
	if err := json.Unmarshal([]byte(raw), &rec); err != nil {
		log.Error("dropping a call record no node can read", "error", err)
		p.forget(id, log)
		return
	}
	if rec.State.Customer.CallID == "" || rec.State.Stream.CallID == "" {
		log.Error("dropping a call record with no dialog to continue")
		p.forget(id, log)
		return
	}

	resumed, err := p.resumer.ResumeTrunkCall(ctx, rec.Ref)
	if errors.Is(err, store.ErrNoSIPTrunk) || errors.Is(err, store.ErrNoStreamTrunk) {
		log.Error("dropping a call through a customer's sip trunk whose trunk is gone", "error", err)
		p.forget(id, log)
		return
	}
	if err != nil {
		// The claim lapses unrenewed, and the next scan tries again.
		log.Warn("could not open the credentials of a call through a customer's sip trunk to take it over", "error", err)
		return
	}
	ended := resumed.Ended
	if ended == nil {
		ended = func(phone.CallStatus) {}
	}
	trunk := resumed.Trunk
	changed := make(chan struct{}, 1)
	cfg := sipbridge.ResumeConfig{
		CustomerTrunk: sipbridge.CustomerTrunk{
			Host: trunk.Host, Port: trunk.Port, Transport: trunk.Transport, Username: trunk.Username,
			Password: trunk.Password, LateOffer: trunk.LateOffer, Codecs: trunk.Codecs,
		},
		Stream: sipbridge.StreamTrunk{
			URI: resumed.Bridge.URI, Username: resumed.Bridge.Username, Password: resumed.Bridge.Password,
		},
		OnChange: notify(changed),
		Logger:   log,
	}.WithDefaults()
	if err := cfg.Validate(); err != nil {
		log.Error("dropping a call through a customer's sip trunk no node can continue", "error", err)
		p.forget(id, log)
		return
	}
	// Recorded before any re-INVITE goes out: a node that took the call next with the old CSeqs
	// would send the ones this node sent, which the peers refuse with a 500.
	sent := rec
	sent.State.Customer.LocalCSeq += cfg.CSeqJump
	sent.State.Stream.LocalCSeq += cfg.CSeqJump
	record, err := json.Marshal(sent)
	if err != nil {
		log.Warn("could not record the CSeqs a takeover sends", "error", err)
		return
	}
	written, err := advanceScript.Exec(ctx, p.redis, []string{p.keys.record(id), p.keys.owner(id)},
		[]string{p.node, string(record), ms(recordFor)}).AsInt64()
	if err == nil && written == -1 {
		log.Info("a call through a customer's sip trunk ended before it was taken over")
		p.forget(id, log)
		return
	}
	if err != nil || written == 0 {
		// The claim lapses unrenewed, and the next scan tries again.
		log.Warn("could not record the CSeqs a takeover sends", "error", err)
		return
	}

	bridged, err := p.resume(ctx, cfg, rec.State)
	if err != nil {
		// The call was answered, and resuming it sent both sides a BYE.
		log.Warn("could not continue a call through a customer's sip trunk another node held", "error", err)
		ended(statusOf(nil))
		p.forget(id, log)
		return
	}
	log.Info("continued a call through a customer's sip trunk another node held")
	// Adopted before this returns, so a Close waiting for the scan hands the call on.
	held := p.adopt(id, bridged, &rec.Ref, true, log)
	if held == nil {
		return
	}
	p.running.Go(func() { p.hold(id, held, changed, log, func() { ended(statusOf(nil)) }) })
}

// claim records a call this node holds and claims it, and says false when another node holds
// it.
func (p *Provider) claim(id string, held *heldCall) (bool, error) {
	record, err := json.Marshal(callRecord{ID: id, Ref: *held.ref, State: held.call.State()})
	if err != nil {
		return false, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), redisTimeout)
	defer cancel()
	lapse := time.Now().Add(p.holdFor)
	claimed, err := claimScript.Exec(ctx, p.redis, []string{p.keys.record(id), p.keys.owner(id), p.keys.calls()},
		[]string{p.node, string(record), ms(recordFor), ms(p.holdFor), strconv.FormatInt(lapse.UnixMilli(), 10), id}).AsInt64()
	if err == nil && claimed == 1 {
		// Not shared yet: adopt adds the call to the ones renewed only after this.
		held.lapse = lapse
	}
	return claimed == 1, err
}

// rewrite writes a held call's record again after its dialogs changed.
func (p *Provider) rewrite(id string, held *heldCall, log *slog.Logger) {
	// So a renewal that read the call's state later is not overwritten by this older one.
	held.mu.Lock()
	defer held.mu.Unlock()
	if held.ended {
		return
	}
	record, err := json.Marshal(callRecord{ID: id, Ref: *held.ref, State: held.call.State()})
	if err != nil {
		log.Warn("could not record a call's changed dialogs", "error", err)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), redisTimeout)
	defer cancel()
	err = writeScript.Exec(ctx, p.redis, []string{p.keys.record(id), p.keys.owner(id)},
		[]string{p.node, string(record), ms(recordFor)}).Error()
	if err != nil {
		log.Warn("could not record a call's changed dialogs", "error", err)
	}
}

// forget deletes an ended call's keys.
func (p *Provider) forget(id string, log *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), redisTimeout)
	defer cancel()
	err := forgetScript.Exec(ctx, p.redis, []string{p.keys.record(id), p.keys.owner(id), p.keys.calls()},
		[]string{p.node, id}).Error()
	if err != nil {
		log.Warn("could not delete an ended call through a customer's sip trunk", "error", err)
	}
}

// handOver gives up this node's claim on a call, tells the other nodes, and lets the call go
// once one of them holds it. Until then the call keeps relaying, so a BYE still ends it here.
// A call let go that no node took sets notTaken first, so Close sees it once the call ended.
func (p *Provider) handOver(ctx context.Context, id string, held *heldCall) {
	log := p.logger.With("vendor", phone.SIPTrunkVendor, "vendor_call_id", id)
	record, err := json.Marshal(callRecord{ID: id, Ref: *held.ref, State: held.call.State()})
	if err != nil {
		log.Warn("could not record a call to hand it to another node", "error", err)
		p.notTaken.Store(true)
		held.call.Release()
		return
	}
	gaveUp, err := giveUpScript.Exec(ctx, p.redis, []string{p.keys.record(id), p.keys.owner(id), p.keys.calls()},
		[]string{p.node, string(record), ms(recordFor), id}).AsInt64()
	if err != nil {
		// The claim lapses unrenewed, and the next node to scan takes the call from there.
		log.Warn("could not offer a call through a customer's sip trunk to another node", "error", err)
		p.notTaken.Store(true)
		held.call.Release()
		return
	}
	if gaveUp == 0 {
		// Another node took it already, or this node's claim lapsed and a scan will.
		held.call.Release()
		return
	}
	if err := p.redis.Do(ctx, p.redis.B().Publish().Channel(p.keys.free()).Message(id).Build()).Error(); err != nil {
		log.Warn("could not tell other nodes a call is free; they find it by scanning", "error", err)
	}

	ticker := time.NewTicker(handoffPoll)
	defer ticker.Stop()
	for {
		select {
		case <-held.call.Done():
			return
		case <-ctx.Done():
			log.Warn("no node took a call through a customer's sip trunk before this one stopped; leaving it for the next to scan")
			p.notTaken.Store(true)
			held.call.Release()
			return
		case <-ticker.C:
			owner, err := p.redis.Do(ctx, p.redis.B().Get().Key(p.keys.owner(id)).Build()).ToString()
			if err == nil && owner != p.node {
				log.Info("handed a call through a customer's sip trunk to another node", "node", owner)
				held.call.Release()
				return
			}
		}
	}
}

func ms(d time.Duration) string { return strconv.FormatInt(d.Milliseconds(), 10) }
