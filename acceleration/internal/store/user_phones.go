package store

import (
	"context"
	"errors"
	"fmt"
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
	UpdatedAt time.Time `bun:"updated_at,notnull"`
}

// LinkUserPhone links a user to a number, in place of any number the user had, and points the
// user's and the number's contact map rows of each agent at one omni-channel (joinLinked). A
// number another user of the app holds is ErrPhoneTaken. Linking a pair again changes nothing.
func (s *Store) LinkUserPhone(ctx context.Context, link *UserPhone) error {
	if link.CustomerID == "" || link.UserID == "" || link.Phone == "" {
		return stack.Wrap(errors.New("store: a customer, a user and a number are required"))
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	link.CreatedAt, link.UpdatedAt = now, now
	link.StreamAppPK = s.linkApp(link.StreamAppPK)
	_, err := s.db.NewInsert().Model(link).
		On("CONFLICT (customer_id, stream_app_pk, user_id) DO UPDATE").
		Set("phone = EXCLUDED.phone").
		Set("updated_at = EXCLUDED.updated_at").
		Returning("*").
		Exec(ctx)
	if isViolation(err, uniqueViolation) {
		return stack.Wrap(ErrPhoneTaken)
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: link user phone: %w", err))
	}
	// After the link is committed, not in its transaction: a contact row MapContact makes at the
	// same time either is committed before this reads it, or reads the link itself once it is
	// made (MapContact), so no row is left out of the pair.
	return s.joinLinked(ctx, *link)
}

// UnlinkUserPhone removes the user's link to a number, if any. The contact rows already
// pointed at one omni-channel stay so: their cards are one person's history.
func (s *Store) UnlinkUserPhone(ctx context.Context, customerID string, app int64, userID string) error {
	_, err := s.db.NewDelete().Model((*UserPhone)(nil)).
		Where("customer_id = ?", customerID).Where("stream_app_pk = ?", s.linkApp(app)).Where("user_id = ?", userID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: unlink user phone: %w", err))
	}
	return nil
}

// linkOf is the link a contact map address is one half of, in the app the row is pinned to. A
// row of another kind, or with no link, has none.
func (s *Store) linkOf(ctx context.Context, entry *ContactMapEntry) (UserPhone, bool, error) {
	column := map[string]string{ContactStreamUser: "user_id", ContactPhone: "phone"}[entry.Kind]
	if column == "" {
		return UserPhone{}, false, nil
	}
	var links []UserPhone
	err := s.db.NewSelect().Model(&links).
		Where("customer_id = ?", entry.CustomerID).
		Where("? = ?", bun.Ident(column), entry.Address).
		Where("stream_app_pk = ?", s.linkApp(entry.StreamAppPK)).
		Limit(1).
		Scan(ctx)
	if err != nil {
		return UserPhone{}, false, stack.Wrap(fmt.Errorf("store: link of a contact: %w", err))
	}
	if len(links) == 0 {
		return UserPhone{}, false, nil
	}
	return links[0], true, nil
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

// joinLinked points the Stream user's row and the number's row of the link at one omni-channel,
// for each agent config of the customer: the older one, whose oldest row
// was made first, and its pin with it. No card moves: a card is read by its message id
// (omnichannel.summaryOf), and the cards of a person are those of every row pointing at their
// omni-channel (EpisodeCards). Two joins at once, a link's and MapContact's, pick the same
// omni-channel, since what makes one older never changes.
func (s *Store) joinLinked(ctx context.Context, link UserPhone) error {
	deployment := s.streamPins().deployment()
	_, err := s.db.NewRaw(`WITH pair AS (
	SELECT cm.id, cm.agent_config_id, cm.conversation_id, cm.stream_app_pk FROM contact_map AS cm
	WHERE cm.customer_id = ?0
		AND ((cm.kind = ?1 AND cm.address = ?2) OR (cm.kind = ?3 AND cm.address = ?4))
		AND COALESCE(NULLIF(cm.stream_app_pk, ?5), 0) = ?6
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
SET conversation_id = oldest.conversation_id, stream_app_pk = oldest.stream_app_pk, updated_at = ?7
FROM pair JOIN oldest USING (agent_config_id)
WHERE cm.id = pair.id AND cm.conversation_id <> oldest.conversation_id`,
		link.CustomerID, ContactStreamUser, link.UserID, ContactPhone, link.Phone,
		deployment, link.StreamAppPK, time.Now().UTC()).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: join linked contacts: %w", err))
	}
	return nil
}
