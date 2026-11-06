package store

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrPhoneTaken says the number is linked to another user of the same Stream app: a number is
// one person, so it is never two users of one app.
var ErrPhoneTaken = errors.New("store: the number is linked to another user of the app")

// UserPhone says a Stream user of a customer's app and a phone number are one person
// (20261212100000_user_phones.sql).
type UserPhone struct {
	bun.BaseModel `bun:"table:user_phones,alias:up"`

	CustomerID string `bun:"customer_id,pk"`
	// StreamAppPK is the app the user is a user of, 0 for the deployment's own.
	StreamAppPK int64  `bun:"stream_app_pk,pk"`
	UserID      string `bun:"user_id,pk"`
	// Phone is the number in E.164.
	Phone     string    `bun:"phone,notnull"`
	CreatedAt time.Time `bun:"created_at,notnull"`
	// UpdatedAt is when the user was linked to Phone: the same number again leaves it, so what
	// the number said since the link stays the user's (EpisodeCards).
	UpdatedAt time.Time `bun:"updated_at,notnull"`
	// History is how much of the number's past a link request gives the user. Nil keeps what the
	// same number had, and gives a new number's link its own time (user_phone_history).
	History *PhoneHistory `bun:"-"`
}

// PhoneHistory is how much of a linked number's past its user reads: everything it ever said
// (All), or what it said from Since on. Neither is from the link request on.
type PhoneHistory struct {
	All   bool
	Since time.Time
}

// historyJoin joins user_phone_history as h to a link up; historyBoundary is then the time
// before which the number's past is not the user's, NULL for none. A link with no row, or a
// row written for another link time, which a router that knows no such row leaves when it moves
// the link to another number and back, is bounded by the link's time.
const (
	historyJoin = `LEFT JOIN user_phone_history AS h ON h.customer_id = up.customer_id
	AND h.stream_app_pk = up.stream_app_pk AND h.user_id = up.user_id`
	historyBoundary = `CASE WHEN h.linked_at IS DISTINCT FROM up.updated_at THEN up.updated_at ELSE h.since END`
)

// setAsideRead are the rows a link set aside (retireHistory) of the number linked to the person
// of an omni-channel, under one agent config, when the link's boundary is before its own time:
// all of the number's past, or from an earlier time on. The link is found as linkOf finds it,
// from the omni-channel's Stream user rows and number rows; the set-aside rows by their number
// (contact_map_retired), in the link's app. The rows stay where they are: a router that knows
// no boundary reads none of them. Arguments: ?0 customer, ?1 agent config, ?2 omni-channel,
// ?3 the deployment's app, ?4 ContactStreamUser, ?5 ContactPhone.
const setAsideRead = `SELECT cm.* FROM (
	SELECT up.* FROM contact_map AS me JOIN user_phones AS up ON up.customer_id = me.customer_id
		AND up.user_id = substr(me.address, strpos(me.address, ':') + 1)
		AND (me.address = up.stream_app_pk || ':' || up.user_id OR up.stream_app_pk = 0 AND me.address = ?3 || ':' || up.user_id)
	WHERE me.customer_id = ?0 AND me.agent_config_id = ?1 AND me.conversation_id = ?2 AND me.kind = ?4
	UNION
	SELECT up.* FROM contact_map AS me JOIN user_phones AS up ON up.customer_id = me.customer_id
		AND up.phone = me.address AND up.stream_app_pk = COALESCE(NULLIF(me.stream_app_pk, ?3), 0)
	WHERE me.customer_id = ?0 AND me.agent_config_id = ?1 AND me.conversation_id = ?2 AND me.kind = ?5
) AS up ` + historyJoin + `
JOIN contact_map AS cm ON cm.customer_id = up.customer_id AND cm.address LIKE '` + retiredPrefix + `%'
	AND ` + reachedAt + ` = up.phone
WHERE COALESCE(` + historyBoundary + ` < up.updated_at, true)
	AND cm.agent_config_id = ?1 AND cm.kind = ?5 AND COALESCE(NULLIF(cm.stream_app_pk, ?3), 0) = up.stream_app_pk`

// userLinkLockSeed is the second argument to hashtextextended for the lock a user's link is
// changed and joined under (lockLink), as definitionLockSeed is a revision's. 2701 is this
// lock's issue, AI-2701; the value only has to differ from the other seeds (731, 833).
const userLinkLockSeed = 2701

// LinkUserPhone links a user to a number, in place of any number the user had, and points the
// user's and the number's contact map rows of each agent at one omni-channel (joinLinked). The
// number the user had before keeps none of the user's history: its rows that share the user's
// omni-channel get one of their own (detachLinked). The new number brings none of its own: its
// rows that hold episodes are set aside (retireHistory), so the user reads only what it says
// after the link, unless the link's History gives the user more of it (writeHistory, EpisodeCards).
// A number another user of the app holds is ErrPhoneTaken. Linking a pair again with no History
// changes nothing.
func (s *Store) LinkUserPhone(ctx context.Context, link *UserPhone) error {
	if link.CustomerID == "" || link.UserID == "" || link.Phone == "" {
		return stack.Wrap(errors.New("store: a customer, a user and a number are required"))
	}
	link.StreamAppPK = s.linkApp(link.StreamAppPK)
	err := s.lockLink(ctx, *link, func(ctx context.Context, tx bun.Tx) error {
		var had []string
		if err := tx.NewSelect().Model((*UserPhone)(nil)).Column("phone").
			Where("customer_id = ?", link.CustomerID).Where("stream_app_pk = ?", link.StreamAppPK).
			Where("user_id = ?", link.UserID).Scan(ctx, &had); err != nil {
			return stack.Wrap(fmt.Errorf("store: read user phone: %w", err))
		}
		// The link's time is Postgres's, as an SMS episode's start is (the inbound queue's
		// created_at), so an SMS queued after the link is never dated before it by the skew of
		// two clocks (EpisodeCards).
		_, err := tx.NewInsert().Model(link).
			Value("created_at", "statement_timestamp()").
			Value("updated_at", "statement_timestamp()").
			On("CONFLICT (customer_id, stream_app_pk, user_id) DO UPDATE").
			Set("phone = EXCLUDED.phone").
			Set("updated_at = CASE WHEN up.phone = EXCLUDED.phone THEN up.updated_at ELSE EXCLUDED.updated_at END").
			Returning("*").
			Exec(ctx)
		if isViolation(err, uniqueViolation) {
			return stack.Wrap(ErrPhoneTaken)
		}
		if err != nil {
			return stack.Wrap(fmt.Errorf("store: link user phone: %w", err))
		}
		same := len(had) > 0 && had[0] == link.Phone
		if err := writeHistory(ctx, tx, *link, same); err != nil {
			return err
		}
		if same {
			return nil
		}
		if err := s.retireHistory(ctx, tx, *link); err != nil {
			return err
		}
		if len(had) == 0 {
			return nil
		}
		return s.detachLinked(ctx, tx, *link, had[0])
	})
	if err != nil {
		return err
	}
	// After the link is committed, not in its transaction: a contact row MapContact makes at the
	// same time either is committed before this reads it, or reads the link itself once it is
	// made (MapContact), so no row is left out of the pair.
	return s.joinLinked(ctx, *link)
}

// writeHistory stores the boundary a link request gives the number's past: none for All, Since,
// or the request's own time, which on a new number is the link's, for the link's time as written
// (linked_at). The same number with no word on its history keeps the boundary it had. In tx,
// under the link's lock, after the link is written.
func writeHistory(ctx context.Context, tx bun.Tx, link UserPhone, same bool) error {
	if link.History == nil && same {
		return nil
	}
	var since any = bun.Safe("up.updated_at")
	switch {
	case link.History == nil:
	case link.History.All:
		since = nil
	case !link.History.Since.IsZero():
		since = link.History.Since
	case same:
		since = bun.Safe("statement_timestamp()")
	}
	_, err := tx.NewRaw(`INSERT INTO user_phone_history (customer_id, stream_app_pk, user_id, linked_at, since)
SELECT up.customer_id, up.stream_app_pk, up.user_id, up.updated_at, ?::timestamptz FROM user_phones AS up
WHERE up.customer_id = ? AND up.stream_app_pk = ? AND up.user_id = ?
ON CONFLICT (customer_id, stream_app_pk, user_id) DO UPDATE SET linked_at = EXCLUDED.linked_at, since = EXCLUDED.since`,
		since, link.CustomerID, link.StreamAppPK, link.UserID).Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: link user phone: history: %w", err))
	}
	return nil
}

// LinkedPhone is a user's link as stored: the number, and from when on its past is the user's,
// nil for all of it. A user with no link has none.
func (s *Store) LinkedPhone(ctx context.Context, customerID string, app int64, userID string) (string, *time.Time, bool, error) {
	var rows []struct {
		Phone string     `bun:"phone"`
		Since *time.Time `bun:"since"`
	}
	err := s.db.NewRaw(`SELECT up.phone, `+historyBoundary+` AS since FROM user_phones AS up `+historyJoin+`
WHERE up.customer_id = ? AND up.stream_app_pk = ? AND up.user_id = ?`, customerID, s.linkApp(app), userID).Scan(ctx, &rows)
	if err != nil {
		return "", nil, false, stack.Wrap(fmt.Errorf("store: linked phone: %w", err))
	}
	if len(rows) == 0 {
		return "", nil, false, nil
	}
	return rows[0].Phone, rows[0].Since, true, nil
}

// ThreadHistorySince is the time before which a thread channel's messages are not restored to a
// session: the boundary of the link of a number an episode of the thread is with, the newest
// first. ok is false when every message is: no such episode, no such link, or a link that gives
// the user all of the number's past. A row a link set aside is read as its number (reachedAt).
func (s *Store) ThreadHistorySince(ctx context.Context, customerID, threadChannel string) (time.Time, bool, error) {
	var since []struct {
		At *time.Time `bun:"since"`
	}
	err := s.db.NewRaw(`SELECT `+historyBoundary+` AS since FROM episodes AS ep
JOIN contact_map AS cm ON cm.id = ep.contact_id
JOIN user_phones AS up ON up.customer_id = cm.customer_id AND up.phone = `+reachedAt+`
	AND up.stream_app_pk = COALESCE(NULLIF(cm.stream_app_pk, ?), 0)
`+historyJoin+`
WHERE ep.customer_id = ? AND ep.thread_channel = ? AND ep.session_id IS NULL AND cm.customer_id = ? AND cm.kind = ?
ORDER BY ep.started_at DESC LIMIT 1`, s.streamPins().deployment(), customerID, threadChannel, customerID, ContactPhone).Scan(ctx, &since)
	if err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: thread history since: %w", err))
	}
	if len(since) == 0 || since[0].At == nil {
		return time.Time{}, false, nil
	}
	return *since[0].At, true, nil
}

// UnlinkUserPhone removes the user's link to a number, if any. The number keeps none of the
// user's history: its rows that share the user's omni-channel get one of their own
// (detachLinked), so whoever has the number next reads none of the user's cards.
func (s *Store) UnlinkUserPhone(ctx context.Context, customerID string, app int64, userID string) error {
	link := UserPhone{CustomerID: customerID, StreamAppPK: s.linkApp(app), UserID: userID}
	return s.lockLink(ctx, link, func(ctx context.Context, tx bun.Tx) error {
		var had []string
		_, err := tx.NewDelete().Model((*UserPhone)(nil)).
			Where("customer_id = ?", link.CustomerID).Where("stream_app_pk = ?", link.StreamAppPK).Where("user_id = ?", link.UserID).
			Returning("phone").
			Exec(ctx, &had)
		if err != nil {
			return stack.Wrap(fmt.Errorf("store: unlink user phone: %w", err))
		}
		if len(had) == 0 {
			return nil
		}
		return s.detachLinked(ctx, tx, link, had[0])
	})
}

// lockLink runs fn in a transaction that holds the lock of one user's link, so the changes
// and joins of one link take turns: two PUTs at once for one user, or a PUT and a first
// sighting, each find the link the one before left (joinLinked reads it under the lock).
func (s *Store) lockLink(ctx context.Context, link UserPhone, fn func(context.Context, bun.Tx) error) error {
	return s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, ?))",
			link.CustomerID+"/"+strconv.FormatInt(link.StreamAppPK, 10)+"/"+link.UserID, userLinkLockSeed); err != nil {
			return stack.Wrap(fmt.Errorf("store: lock user phone: %w", err))
		}
		return fn(ctx, tx)
	})
}

// linkOf is the link a contact map address is one half of: a Stream user's in the app its
// address names, a number's in the app the row is pinned to. A row of another kind, or with no
// link, has none.
func (s *Store) linkOf(ctx context.Context, entry *ContactMapEntry) (UserPhone, bool, error) {
	query := s.db.NewSelect().Model((*UserPhone)(nil)).Where("customer_id = ?", entry.CustomerID)
	switch entry.Kind {
	case ContactStreamUser:
		app, userID, ok := splitStreamUserAddress(entry.Address)
		if !ok {
			return UserPhone{}, false, nil
		}
		query = query.Where("user_id = ?", userID).Where("stream_app_pk = ?", s.linkApp(app))
	case ContactPhone:
		query = query.Where("phone = ?", entry.Address).Where("stream_app_pk = ?", s.linkApp(entry.StreamAppPK))
	default:
		return UserPhone{}, false, nil
	}
	var links []UserPhone
	if err := query.Limit(1).Scan(ctx, &links); err != nil {
		return UserPhone{}, false, stack.Wrap(fmt.Errorf("store: link of a contact: %w", err))
	}
	if len(links) == 0 {
		return UserPhone{}, false, nil
	}
	return links[0], true, nil
}

// StreamUserAddress is a stream_user row's address, «<app pk>:<user id>»: a user id is a person
// only within its Stream app. omnichannel.StreamUser writes rows with it, and a link finds
// them with it.
func StreamUserAddress(app int64, userID string) string {
	return strconv.FormatInt(app, 10) + ":" + userID
}

// splitStreamUserAddress reads a stream_user row's address back into its app and user id.
func splitStreamUserAddress(address string) (int64, string, bool) {
	pin, userID, ok := strings.Cut(address, ":")
	app, err := strconv.ParseInt(pin, 10, 64)
	if !ok || err != nil || userID == "" {
		return 0, "", false
	}
	return app, userID, true
}

// linkApp is the app a pin names as a link keeps it: 0 for the deployment's own, which a row
// may hold as NULL, as 0 or as the deployment app's id. The message hook pins a chat row from
// the deployment's own app to its id (streamapp.Verifier), while a request acting in it is
// pinned 0 (streamapp.Identity), so both name one app here.
func (s *Store) linkApp(pin int64) int64 {
	if deployment := s.streamPins().deployment(); deployment != 0 && pin == deployment {
		return 0
	}
	return pin
}

// streamUserAddresses are the addresses a link's Stream user has in the contact map: in the
// deployment's own app, as 0 and as the app's id, which the message hook names it by once it is
// known (linkApp).
func (s *Store) streamUserAddresses(link UserPhone) []string {
	addresses := []string{StreamUserAddress(link.StreamAppPK, link.UserID)}
	if deployment := s.streamPins().deployment(); link.StreamAppPK == 0 && deployment != 0 {
		addresses = append(addresses, StreamUserAddress(deployment, link.UserID))
	}
	return addresses
}

// joinLinked points the Stream user's row and the number's row of the user's link at one
// omni-channel, for each agent config of the customer: the older one, whose oldest row was
// made first, and its pin with it. The number is the link as stored, read under the link's
// lock, not link.Phone: a join that comes after another PUT of the user joins the number that
// PUT left. No card moves: a card is read by its message id (omnichannel.summaryOf), and the
// cards of a person are those of every row pointing at their omni-channel (EpisodeCards). Two
// joins of one link take turns under its lock, so the second finds the first's rows on one
// omni-channel already. Each half of the pair is found for each of the customer's agent
// configs by the whole of contact_map_address, and born by contact_map_conversation, so a join
// reads the pair's rows, not every row of the customer: 0.6 ms against 52 ms with one OR'd
// pair over the customer's range, at 300k rows of one customer among 900k (EXPLAIN ANALYZE,
// AI-2701 review).
func (s *Store) joinLinked(ctx context.Context, link UserPhone) error {
	deployment := s.streamPins().deployment()
	return s.lockLink(ctx, link, func(ctx context.Context, tx bun.Tx) error {
		_, err := tx.NewRaw(`WITH link AS (
	SELECT up.phone FROM user_phones AS up WHERE up.customer_id = ?0 AND up.stream_app_pk = ?5 AND up.user_id = ?7
), configs AS (
	SELECT ac.id AS agent_config_id FROM agent_configs AS ac WHERE ac.customer_id = ?0
), pair AS (
	SELECT cm.id, cm.agent_config_id, cm.conversation_id, cm.stream_app_pk FROM configs JOIN contact_map AS cm USING (agent_config_id)
	WHERE cm.customer_id = ?0 AND cm.kind = ?1 AND cm.address IN (?2) AND COALESCE(NULLIF(cm.stream_app_pk, ?4), 0) = ?5
	UNION ALL
	SELECT cm.id, cm.agent_config_id, cm.conversation_id, cm.stream_app_pk FROM configs JOIN contact_map AS cm USING (agent_config_id)
	WHERE cm.customer_id = ?0 AND cm.kind = ?3 AND cm.address = (SELECT phone FROM link) AND COALESCE(NULLIF(cm.stream_app_pk, ?4), 0) = ?5
), born AS (
	SELECT o.agent_config_id, o.conversation_id, min(o.created_at) AS at FROM contact_map AS o
	WHERE o.customer_id = ?0 AND (o.agent_config_id, o.conversation_id) IN (SELECT agent_config_id, conversation_id FROM pair)
	GROUP BY o.agent_config_id, o.conversation_id
), oldest AS (
	SELECT DISTINCT ON (pair.agent_config_id) pair.agent_config_id, pair.conversation_id, pair.stream_app_pk
	FROM pair JOIN born USING (agent_config_id, conversation_id)
	ORDER BY pair.agent_config_id, born.at, pair.conversation_id
)
UPDATE contact_map AS cm
SET conversation_id = oldest.conversation_id, stream_app_pk = oldest.stream_app_pk, updated_at = ?6
FROM pair JOIN oldest USING (agent_config_id)
WHERE cm.id = pair.id AND cm.conversation_id <> oldest.conversation_id`,
			link.CustomerID, ContactStreamUser, bun.In(s.streamUserAddresses(link)), ContactPhone,
			deployment, link.StreamAppPK, time.Now().UTC(), link.UserID).
			Exec(ctx)
		if err != nil {
			return stack.Wrap(fmt.Errorf("store: join linked contacts: %w", err))
		}
		return nil
	})
}

// detachLinked points each row of the number a user is no longer linked to that shares the
// user's omni-channel, under each agent config, at a new omni-channel of its own, in its own
// pin's app: the number keeps no history of the user. Its episodes go with it (episodes.contact_id),
// so the number keeps its own cards, read by their message id where they were written
// (omnichannel.summaryOf); the user keeps theirs. The id is omnichannel.Open's form,
// agent:omni-<uuid>. In tx, under the link's lock.
func (s *Store) detachLinked(ctx context.Context, tx bun.Tx, link UserPhone, phone string) error {
	_, err := tx.NewRaw(`UPDATE contact_map AS cm
SET conversation_id = 'agent:omni-' || gen_random_uuid(), updated_at = ?6
FROM contact_map AS su
WHERE su.customer_id = ?0 AND su.agent_config_id IN (SELECT ac.id FROM agent_configs AS ac WHERE ac.customer_id = ?0)
	AND su.kind = ?1 AND su.address IN (?2)
	AND cm.customer_id = ?0 AND cm.agent_config_id = su.agent_config_id AND cm.kind = ?3 AND cm.address = ?4
	AND COALESCE(NULLIF(cm.stream_app_pk, ?5), 0) = ?7
	AND cm.conversation_id = su.conversation_id`,
		link.CustomerID, ContactStreamUser, bun.In(s.streamUserAddresses(link)), ContactPhone, phone,
		s.streamPins().deployment(), time.Now().UTC(), link.StreamAppPK).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: detach unlinked contacts: %w", err))
	}
	return nil
}

// retiredPrefix starts the address of a number's row a link set aside (retireHistory),
// retired:<row id>:<number>. No E.164 number starts with it, so no sighting, read or join
// finds the row by the number again.
const retiredPrefix = "retired:"

// reachedAt is a row's address as the person is reached at: the number of a row a link set
// aside (retireHistory), any other row's address as it is. What still talks to the person on
// a thread, call or card the row holds reads it (ThreadContact, the closed episodes' and the
// cards' contact_address), so a reply's opt-out, sandbox and limit checks and a call's caller
// check name the number. Example: retired:3f2a…:+15550100 is reached at +15550100.
const reachedAt = `regexp_replace(cm.address, '^` + retiredPrefix + `[^:]*:', '')`

// retireHistory sets aside each row of a number newly linked to a user that holds an episode,
// under each agent config of the customer that writes episode cards, on or record
// (AgentConfig.WritesCards), in the link's app: whoever
// texted or called from the number before (a former holder, or the user before the customer
// vouched for them) is not proven to be the user. The row keeps its omni-channel and its
// episodes, so their cards stay where they were written and are closed there, but its address
// becomes retired:<row id>:<number>, which no lookup by the number matches; what talks to the
// person on its thread, call or cards still reads the number (reachedAt). The number's next
// episode makes a row of its own, which joins the user (joinPartner). A row with no episode has nothing
// to set aside and joins as it is (joinLinked); a config with episode_cards off keeps its rows
// as they were, and EpisodeCards' link filter keeps their episodes from the user. Record
// counts as on: it writes the same episodes, which a session reads once the config turns on.
// In tx, under the link's lock.
func (s *Store) retireHistory(ctx context.Context, tx bun.Tx, link UserPhone) error {
	_, err := tx.NewRaw(`UPDATE contact_map AS cm
SET address = ?4 || cm.id || ':' || cm.address, updated_at = ?5
WHERE cm.customer_id = ?0
	AND cm.agent_config_id IN (SELECT ac.id FROM agent_configs AS ac WHERE ac.customer_id = ?0 AND (ac.episode_cards OR ac.episode_cards_record))
	AND cm.kind = ?1 AND cm.address = ?2
	AND COALESCE(NULLIF(cm.stream_app_pk, ?3), 0) = ?6
	AND EXISTS (SELECT 1 FROM episodes AS ep WHERE ep.contact_id = cm.id)`,
		link.CustomerID, ContactPhone, link.Phone, s.streamPins().deployment(), retiredPrefix, time.Now().UTC(),
		link.StreamAppPK).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: retire a linked number's history: %w", err))
	}
	return nil
}
