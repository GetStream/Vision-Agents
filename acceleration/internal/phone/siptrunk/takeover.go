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
	// PostDialDelay is what the node that dialled the call measured, for whichever node reports
	// its end.
	PostDialDelay time.Duration
}

// keys names the Redis keys: a call's record, its owner, the set of calls scored by when
// each one's claim lapses, the channel a stopping node offers its calls on, and the set of
// nodes that take calls over, scored by when each one's word lapses.
type keys string

func (k keys) record(id string) string { return string(k) + ":call:" + id }
func (k keys) owner(id string) string  { return string(k) + ":owner:" + id }
func (k keys) calls() string           { return string(k) + ":calls" }
func (k keys) free() string            { return string(k) + ":free" }
func (k keys) nodes() string           { return string(k) + ":nodes" }

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

// giveBackScript lets go of ARGV[1]'s claim on a call it could not take over, and marks the
// call lapsing at ARGV[2], unless it was forgotten meanwhile.
// KEYS: owner, calls. ARGV: node, lapse ms, id.
var giveBackScript = rueidis.NewLuaScript(`
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
redis.call('DEL', KEYS[1])
redis.call('ZADD', KEYS[2], 'XX', ARGV[2], ARGV[3])
return 1
`)

// reclaimScript wins back a call ARGV[1] gave up, while no node holds it and no node says it
// takes calls over, so the call can be hung up rather than left with no node to relay it. It
// says 1 too when ARGV[1] holds it already: an earlier win whose reply was lost.
// KEYS: owner, calls, nodes. ARGV: node, holdFor ms, now ms, id.
var reclaimScript = rueidis.NewLuaScript(`
local owner = redis.call('GET', KEYS[1])
if owner == ARGV[1] then
  return 1
end
if owner then
  return 0
end
if redis.call('ZSCORE', KEYS[2], ARGV[4]) ~= '0' then
  return 0
end
if redis.call('ZCOUNT', KEYS[3], '(' .. ARGV[3], '+inf') > 0 then
  return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
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

// Start renews this node's claims until Close and, with r to open a call's credentials,
// takes over the calls other nodes gave up or stopped renewing until StopTaking. It takes no
// context: a signal starts a drain, through which this node still relays its calls and must
// keep its claims on them. Without Redis it does nothing.
func (p *Provider) Start(r Resumer) {
	if p.redis == nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	p.mu.Lock()
	p.stopLoops = cancel
	p.mu.Unlock()
	p.loops.Go(func() { p.renewLoop(ctx) })
	if r == nil {
		return
	}
	p.resumer = r
	taking, stopTaking := context.WithCancel(ctx)
	p.mu.Lock()
	p.stopTaking = stopTaking
	p.mu.Unlock()
	// Before any call can be offered: a node stopping now must see this one to hand its calls on.
	p.advertise(taking)
	offered := make(chan struct{}, 1)
	p.loops.Go(func() { p.listen(taking, offered) })
	p.loops.Go(func() { p.scanLoop(taking, offered) })
}

// StopTaking stops this node taking calls over and saying it does, while it keeps renewing
// the claims on its own calls: a draining node would only hand them on again at Close. Before
// Start it does nothing.
func (p *Provider) StopTaking() {
	p.mu.Lock()
	stop := p.stopTaking
	p.mu.Unlock()
	if stop == nil {
		return
	}
	p.advertising.Lock()
	defer p.advertising.Unlock()
	// Under the lock advertise checks under, so no word lands after it is taken back here.
	stop()
	ctx, cancel := context.WithTimeout(context.Background(), redisTimeout)
	defer cancel()
	if err := p.redis.Do(ctx, p.redis.B().Zrem().Key(p.keys.nodes()).Member(p.node).Build()).Error(); err != nil {
		p.logger.Warn("could not tell other nodes this one stopped taking calls through customers' sip trunks over", "error", err)
	}
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
	// A call being handed over may look lost once a failed take moved its score off the given-up
	// 0, and claiming it back would undo the offer.
	if h.ended || h.handingOver {
		return
	}
	record, err := json.Marshal(callRecord{ID: id, Ref: *h.ref, State: h.call.State(), PostDialDelay: h.postDialDelay})
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
		p.advertise(ctx)
		p.scan(ctx)
	}
}

// advertise tells stopping nodes that this one takes calls over, until holdFor from now.
func (p *Provider) advertise(ctx context.Context) {
	p.advertising.Lock()
	defer p.advertising.Unlock()
	// Under the lock withdraw and StopTaking take after the node began stopping, so no word
	// lands after theirs.
	if p.ctx.Err() != nil || ctx.Err() != nil {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, redisTimeout)
	defer cancel()
	now := time.Now()
	for _, resp := range p.redis.DoMulti(ctx,
		p.redis.B().Zadd().Key(p.keys.nodes()).ScoreMember().ScoreMember(float64(now.Add(p.holdFor).UnixMilli()), p.node).Build(),
		// Each process is a new node, so the words of the ones gone would pile up.
		p.redis.B().Zremrangebyscore().Key(p.keys.nodes()).Min("-inf").Max(strconv.FormatInt(now.UnixMilli(), 10)).Build(),
	) {
		if err := resp.Error(); err != nil {
			if ctx.Err() == nil {
				p.logger.Warn("could not tell other nodes this one takes calls through customers' sip trunks over", "error", err)
			}
			return
		}
	}
}

// withdraw stops this node saying it takes calls over, and says whether another node still
// does. It says false when Redis cannot tell, so the calls are hung up rather than left with
// no node to relay them.
func (p *Provider) withdraw(ctx context.Context) bool {
	p.advertising.Lock()
	defer p.advertising.Unlock()
	ctx, cancel := context.WithTimeout(ctx, redisTimeout)
	defer cancel()
	resps := p.redis.DoMulti(ctx,
		p.redis.B().Zrem().Key(p.keys.nodes()).Member(p.node).Build(),
		p.redis.B().Zcount().Key(p.keys.nodes()).Min("("+strconv.FormatInt(time.Now().UnixMilli(), 10)).Max("+inf").Build(),
	)
	if err := resps[0].Error(); err != nil {
		p.logger.Warn("could not tell other nodes this one stopped taking calls through customers' sip trunks over", "error", err)
		return false
	}
	others, err := resps[1].AsInt64()
	if err != nil {
		p.logger.Warn("could not find a node to take calls through customers' sip trunks over; hanging them up", "error", err)
		return false
	}
	return others > 0
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
		log.Warn("could not open the credentials of a call through a customer's sip trunk to take it over", "error", err)
		p.giveBack(ctx, id, log)
		return
	}
	ended := resumed.Ended
	if ended == nil {
		ended = func(phone.CallStatus) {}
	}
	report := func() {
		status := statusOf(nil)
		status.PostDialDelay = rec.PostDialDelay
		ended(status)
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
		log.Warn("could not record the CSeqs a takeover sends", "error", err)
		p.giveBack(ctx, id, log)
		return
	}

	bridged, err := p.resume(ctx, cfg, rec.State)
	if err != nil {
		// The call was answered, and resuming it sent both sides a BYE.
		log.Warn("could not continue a call through a customer's sip trunk another node held", "error", err)
		report()
		p.forget(id, log)
		return
	}
	log.Info("continued a call through a customer's sip trunk another node held")
	// Adopted before this returns, so a Close waiting for the scan hands the call on.
	held := p.adopt(id, bridged, &rec.Ref, rec.PostDialDelay, true, log)
	if held == nil {
		return
	}
	p.running.Go(func() { p.hold(id, held, changed, log, report) })
}

// giveBack lets go of a claim won on a call this node failed to take over before it sent
// anything, so a scan tries the call again soon, after the calls that lapsed before then.
func (p *Provider) giveBack(ctx context.Context, id string, log *slog.Logger) {
	lapse := strconv.FormatInt(time.Now().Add(p.renewEvery).UnixMilli(), 10)
	err := giveBackScript.Exec(ctx, p.redis, []string{p.keys.owner(id), p.keys.calls()}, []string{p.node, lapse, id}).Error()
	if err != nil {
		// The claim lapses unrenewed, and a scan tries the call again then.
		log.Warn("could not give back a call through a customer's sip trunk this node did not take over", "error", err)
	}
}

// claim records a call this node holds and claims it, and says false when another node holds
// it.
func (p *Provider) claim(id string, held *heldCall) (bool, error) {
	record, err := json.Marshal(callRecord{ID: id, Ref: *held.ref, State: held.call.State(), PostDialDelay: held.postDialDelay})
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
	record, err := json.Marshal(callRecord{ID: id, Ref: *held.ref, State: held.call.State(), PostDialDelay: held.postDialDelay})
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
// When every node that took calls stopped first, as when all stop together, it wins the call
// back and hangs it up. A call let go that no node took sets notTaken first, so Close sees it
// once the call ended.
func (p *Provider) handOver(ctx context.Context, id string, held *heldCall) {
	log := p.logger.With("vendor", phone.SIPTrunkVendor, "vendor_call_id", id)
	held.mu.Lock()
	held.handingOver = true
	held.mu.Unlock()
	record, err := json.Marshal(callRecord{ID: id, Ref: *held.ref, State: held.call.State(), PostDialDelay: held.postDialDelay})
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
			// Not ctx, which ran out: whether any node is left to take the call is still worth asking.
			lastCtx, cancel := context.WithTimeout(context.Background(), redisTimeout)
			won := p.reclaim(lastCtx, id, log)
			cancel()
			if won {
				p.hangUpUntaken(held, log)
				return
			}
			log.Warn("no node took a call through a customer's sip trunk before this one stopped; leaving it for the next to scan")
			p.notTaken.Store(true)
			held.call.Release()
			return
		case <-ticker.C:
			owner, err := p.redis.Do(ctx, p.redis.B().Get().Key(p.keys.owner(id)).Build()).ToString()
			// Its own owner only after a win-back whose reply was lost, since the claim was given up.
			unowned := rueidis.IsRedisNil(err) || (err == nil && owner == p.node)
			if unowned && p.reclaim(ctx, id, log) {
				p.hangUpUntaken(held, log)
				return
			}
			if err == nil && owner != p.node {
				log.Info("handed a call through a customer's sip trunk to another node", "node", owner)
				held.call.Release()
				return
			}
		}
	}
}

// reclaim wins back a call this node gave up, and says true, when no node holds it and none
// says it takes calls over.
func (p *Provider) reclaim(ctx context.Context, id string, log *slog.Logger) bool {
	won, err := reclaimScript.Exec(ctx, p.redis, []string{p.keys.owner(id), p.keys.calls(), p.keys.nodes()},
		[]string{p.node, ms(p.holdFor), strconv.FormatInt(time.Now().UnixMilli(), 10), id}).AsInt64()
	if err != nil {
		if ctx.Err() == nil {
			log.Warn("could not find whether a node is left to take a call through a customer's sip trunk", "error", err)
		}
		return false
	}
	return won == 1
}

// hangUpUntaken hangs up a call no node is left to take; hold reports its end and deletes it.
func (p *Provider) hangUpUntaken(held *heldCall, log *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), hangupTimeout)
	defer cancel()
	if err := held.call.Hangup(ctx); err != nil {
		log.Warn("could not hang up a call through a customer's sip trunk no node is left to take", "error", err)
		return
	}
	log.Info("hung up a call through a customer's sip trunk because no node is left to take it")
}

// ownsInRedis says whether Redis names this node the call's owner, false when it cannot say.
func (p *Provider) ownsInRedis(id string, log *slog.Logger) bool {
	ctx, cancel := context.WithTimeout(context.Background(), redisTimeout)
	defer cancel()
	owner, err := p.redis.Do(ctx, p.redis.B().Get().Key(p.keys.owner(id)).Build()).ToString()
	if err != nil && !rueidis.IsRedisNil(err) {
		log.Warn("could not read who holds a call through a customer's sip trunk", "error", err)
	}
	return err == nil && owner == p.node
}

func ms(d time.Duration) string { return strconv.FormatInt(d.Milliseconds(), 10) }
