package dlc

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/region"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// wouldRefuseEvery is how often shadow logs the same organization's would-be refusal.
const wouldRefuseEvery = time.Minute

// withinTier refuses what the organization's channel limits do not allow, in enforce; in
// shadow it logs it and allows. Each channel is its own key: a Linq text is checked as
// imessage whichever of iMessage, RCS or SMS carries it.
//
// The countries are checked first, then today's count, then today's recipients, then for a
// call the calls up now. The daily counts fail open with Redis, as the sandbox's do; the
// calls up are read from Postgres and fail open too, logged.
func (g *Gate) withinTier(ctx context.Context, admission routing.Admission, customerID, channel, to string) error {
	if !g.limiting() || admission.Limits == nil || admission.OrganizationID == "" {
		return nil
	}
	limits, ok := admission.Limits.Channels[channel]
	if !ok {
		return nil
	}
	refusal, reason := g.overTier(ctx, admission.OrganizationID, tierNamed(admission.LimitsTier), channel, to, limits)
	if refusal == nil {
		return nil
	}
	if g.mode == config.LimitsShadow {
		if g.logs.due(admission.OrganizationID+"\x00"+channel+"\x00"+reason, wouldRefuseEvery) {
			g.logger.Info("would refuse", "organization", admission.OrganizationID, "app", customerID,
				"channel", channel, "reason", reason, "refusal", refusal)
		}
		return nil
	}
	return refusal
}

// tierNamed is the tier a refusal names: the one the limits are of, or "your organization's".
func tierNamed(tier string) string {
	if tier == "" {
		return "your organization's"
	}
	return "the " + tier
}

// overTier is why the limits refuse a text or call to on channel, with a short reason for the
// logs, or nil. Neither names the number.
func (g *Gate) overTier(ctx context.Context, organization, tier, channel, to string, limits usage.ChannelLimits) (error, string) {
	if len(limits.Countries) > 0 {
		country := region.Of(to)
		if country == "" {
			return stack.Wrap(fmt.Errorf("%w: %s tier allows %s only to %s, and this recipient is in no country",
				ErrRefused, tier, channel, strings.Join(limits.Countries, ", "))), "country"
		}
		if !slices.Contains(limits.Countries, country) {
			return stack.Wrap(fmt.Errorf("%w: %s tier is not allowed to reach %s by %s", ErrRefused, tier, country, channel)), "country"
		}
	}
	if limits.PerDay != nil || limits.RecipientsPerDay != nil {
		today, ok := g.today(ctx, organization, channel, to)
		if ok && limits.PerDay != nil && today.count >= int64(*limits.PerDay) {
			unit := "messages"
			if channel == Voice {
				unit = "minutes"
			}
			return stack.Wrap(fmt.Errorf("%w: %s tier allows %d %s of %s a day, and today's are used up",
				ErrRefused, tier, *limits.PerDay, unit, channel)), "per_day"
		}
		if ok && limits.RecipientsPerDay != nil && !today.reached && today.recipients >= int64(*limits.RecipientsPerDay) {
			return stack.Wrap(fmt.Errorf("%w: %s tier allows %s to %d recipients a day, and today's are all reached; "+
				"the ones already reached can still be", ErrRefused, tier, channel, *limits.RecipientsPerDay)), "recipients_per_day"
		}
	}
	if channel == Voice && limits.Concurrent != nil {
		up, err := g.store.OpenCallLegs(ctx, organization)
		if err != nil {
			g.logger.Error("could not count the calls an organization has up, allowing", "organization", organization, "error", err)
		} else if up >= *limits.Concurrent {
			return stack.Wrap(fmt.Errorf("%w: %s tier allows %d calls at once, and that many are up",
				ErrRefused, tier, *limits.Concurrent)), "concurrent"
		}
	}
	return nil, ""
}

// channelDay is what an organization sent on one channel today, as Redis last said.
type channelDay struct {
	// count is messages, or minutes for voice; recipients how many distinct people were
	// reached, and reached whether the one asked about was.
	count      int64
	recipients int64
	reached    bool
}

// today reads what the organization sent on channel today; ok is false when Redis could not
// say, which allows.
func (g *Gate) today(ctx context.Context, organization, channel, to string) (channelDay, bool) {
	if g.redis == nil {
		return channelDay{}, false
	}
	counts, reached := g.channelKeys(organization, channel)
	answers := g.redis.DoMulti(ctx,
		g.redis.B().Hget().Key(counts).Field(channel).Build(),
		g.redis.B().Sismember().Key(reached).Member(to).Build(),
		g.redis.B().Scard().Key(reached).Build(),
	)
	var day channelDay
	count, err := answers[0].ToString()
	if err != nil && !rueidis.IsRedisNil(err) {
		g.logger.Error("could not read an organization's channel counters, allowing", "error", err)
		return channelDay{}, false
	}
	day.count = parseInt(count)
	if day.reached, err = answers[1].AsBool(); err != nil {
		g.logger.Error("could not read an organization's channel recipients, allowing", "error", err)
		return channelDay{}, false
	}
	if day.recipients, err = answers[2].AsInt64(); err != nil {
		g.logger.Error("could not read an organization's channel recipients, allowing", "error", err)
		return channelDay{}, false
	}
	return day, true
}

// countChannel counts by against the organization's channel today, and to as reached on it
// (by 0 counts only to; an empty to counts only by). With no organization given it is the
// app's, as admission knows it. Nothing is counted while the gate keeps no limits.
func (g *Gate) countChannel(ctx context.Context, customerID, organization, channel, to string, by int64) {
	if !g.limiting() || g.redis == nil || (by <= 0 && to == "") {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()
	if organization == "" && g.money != nil {
		attributed, err := g.money.Attribute(ctx, customerID)
		if err != nil {
			g.logger.Error("could not tell whose channel limits a message counts against", "app", customerID, "error", err)
			return
		}
		organization = attributed.OrganizationID
	}
	if organization == "" {
		return
	}
	counts, reached := g.channelKeys(organization, channel)
	ttl := int64(retention.Seconds())
	commands := make(rueidis.Commands, 0, 4)
	if by > 0 {
		commands = append(commands,
			g.redis.B().Hincrby().Key(counts).Field(channel).Increment(by).Build(),
			g.redis.B().Expire().Key(counts).Seconds(ttl).Build())
	}
	if to != "" {
		commands = append(commands,
			g.redis.B().Sadd().Key(reached).Member(to).Build(),
			g.redis.B().Expire().Key(reached).Seconds(ttl).Build())
	}
	for _, response := range g.redis.DoMulti(ctx, commands...) {
		if err := response.Error(); err != nil {
			g.logger.Error("could not count what an organization sent on a channel", "channel", channel, "error", err)
			return
		}
	}
}

// channelKeys are the organization's counters for today, by channel, and the people reached on
// channel today: a new pair each UTC day, in one cluster slot.
func (g *Gate) channelKeys(organization, channel string) (counts, reached string) {
	counts = "channel:" + time.Now().UTC().Format("20060102") + ":{" + organization + "}"
	return counts, counts + ":" + channel + ":to"
}

// limiting reports whether the gate keeps organizations to their channel limits now.
func (g *Gate) limiting() bool {
	return g != nil && g.limited && g.mode != config.LimitsOff
}

// throttle says whether something keyed is due again, at most once every interval per key.
type throttle struct {
	mu   sync.Mutex
	last map[string]time.Time
}

// maxThrottled is how many keys a throttle remembers before it forgets them all.
const maxThrottled = 10_000

func (t *throttle) due(key string, every time.Duration) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now()
	if t.last == nil || len(t.last) >= maxThrottled {
		t.last = map[string]time.Time{}
	}
	if last, ok := t.last[key]; ok && now.Sub(last) < every {
		return false
	}
	t.last[key] = now
	return true
}
