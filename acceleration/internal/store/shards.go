package store

import (
	"cmp"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// A deployment may keep the rows every part of it agrees on in one database and the calls each
// part runs in a database of its own, the shard's. Calls, sessions and their turns, the
// requests a call made and the rollups of them are the shard's: written on a call's hot path,
// close to where it ran. Everything else is shared.
//
// A deployment of one shard sets one DSN, and every handle below is the same database.

// peerConns is how many connections a router holds to another shard's database. It only reads
// there, for a page somebody is looking at, so it needs far fewer than its own.
const peerConns = 8

// Options says where a store's databases are.
type Options struct {
	// DSN is the database every shard shares.
	DSN string
	// ShardDSN is this shard's, where the calls it runs are written. Empty is DSN.
	ShardDSN string
	// Shards is every shard's DSN by name, this one's included. Reads of calls, sessions and
	// requests cover them all. Empty is this shard's alone.
	Shards map[string]string
}

// OpenShards connects to the shared database, this shard's, and every other shard's.
func OpenShards(options Options) (*Store, error) {
	if options.DSN == "" {
		return nil, errors.New("store: dsn is required")
	}
	db := connect(options.DSN, maxConns)
	s := &Store{db: db, shard: db}
	if options.ShardDSN != "" && options.ShardDSN != options.DSN {
		s.shard = connect(options.ShardDSN, maxConns)
	}
	own := cmp.Or(options.ShardDSN, options.DSN)
	for _, name := range slices.Sorted(maps.Keys(options.Shards)) {
		if dsn := options.Shards[name]; dsn != "" && dsn != own {
			s.peers = append(s.peers, connect(dsn, peerConns))
		}
	}
	return s, nil
}

// Sharded reports whether this shard's calls are kept apart from the shared database.
func (s *Store) Sharded() bool { return s.shard != s.db }

// shards is every shard's database, this one's first.
func (s *Store) shards() []bun.IDB {
	all := make([]bun.IDB, 0, 1+len(s.peers))
	all = append(all, s.shard)
	for _, peer := range s.peers {
		all = append(all, peer)
	}
	return all
}

// everyShard reads every shard's database at once and returns what each answered, this
// shard's first. One shard failing fails the read: a page that silently left a shard out
// would be a list of calls missing some.
func everyShard[T any](ctx context.Context, s *Store, read func(context.Context, bun.IDB) (T, error)) ([]T, error) {
	all := s.shards()
	if len(all) == 1 {
		found, err := read(ctx, all[0])
		if err != nil {
			return nil, err
		}
		return []T{found}, nil
	}
	found := make([]T, len(all))
	failed := make([]error, len(all))
	var wg sync.WaitGroup
	for i, db := range all {
		wg.Go(func() { found[i], failed[i] = read(ctx, db) })
	}
	wg.Wait()
	if err := errors.Join(failed...); err != nil {
		return nil, err
	}
	return found, nil
}

// whicheverShard reads a row from the shard holding it: this one first, where the row almost
// always is, then the others. read answers sql.ErrNoRows for a shard without it, and so does
// whicheverShard when no shard has it.
func whicheverShard[T any](ctx context.Context, s *Store, read func(context.Context, bun.IDB) (T, error)) (T, error) {
	found, err := read(ctx, s.shard)
	if len(s.peers) == 0 || !errors.Is(err, sql.ErrNoRows) {
		return found, err
	}
	answers := make([]T, len(s.peers))
	failed := make([]error, len(s.peers))
	var wg sync.WaitGroup
	for i, peer := range s.peers {
		wg.Go(func() { answers[i], failed[i] = read(ctx, peer) })
	}
	wg.Wait()
	for i := range s.peers {
		if failed[i] == nil {
			return answers[i], nil
		}
	}
	for _, err := range failed {
		if !errors.Is(err, sql.ErrNoRows) {
			return found, err
		}
	}
	return found, sql.ErrNoRows
}

// onEveryShard runs a write on every shard's database and returns how many rows it changed
// across them. It is for a row somebody changes from wherever they are, long after the shard
// that wrote it is known to anybody.
func onEveryShard(ctx context.Context, s *Store, write func(context.Context, bun.IDB) (sql.Result, error)) (int64, error) {
	results, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (int64, error) {
		result, err := write(ctx, db)
		if err != nil {
			return 0, err
		}
		return result.RowsAffected()
	})
	if err != nil {
		return 0, err
	}
	var total int64
	for _, affected := range results {
		total += affected
	}
	return total, nil
}

// merged puts every shard's rows into one list in the order before says, cut to limit when
// limit is above zero. The sort is stable, so rows that tie keep this shard's first.
func merged[T any](lists [][]T, before func(a, b T) int, limit int) []T {
	if len(lists) == 1 {
		return lists[0]
	}
	var all []T
	for _, list := range lists {
		all = append(all, list...)
	}
	slices.SortStableFunc(all, before)
	if limit > 0 && len(all) > limit {
		all = all[:limit]
	}
	return all
}

// distinct keeps the first row of each key. A text session carried on in another shard than
// the one it began in has a row in both, and the first, in an order newest first, is the one
// that says what it is now.
func distinct[T any](rows []T, key func(T) string) []T {
	seen := make(map[string]bool, len(rows))
	kept := rows[:0]
	for _, row := range rows {
		if !seen[key(row)] {
			seen[key(row)] = true
			kept = append(kept, row)
		}
	}
	return kept
}

// closeShards closes every database the store opened past the shared one.
func (s *Store) closeShards() error {
	var failed []error
	if s.Sharded() {
		failed = append(failed, s.shard.Close())
	}
	for _, peer := range s.peers {
		failed = append(failed, peer.Close())
	}
	return stack.Wrap(errors.Join(failed...))
}

// migrateShard brings this shard's database to the same schema as the shared one. Every
// database carries every table, so a deployment can be split, or a shard added, without a
// second migration history to keep in step.
func (s *Store) migrateShard(ctx context.Context) error {
	if !s.Sharded() {
		return nil
	}
	if err := s.migrate(ctx, s.shard); err != nil {
		return fmt.Errorf("store: shard: %w", err)
	}
	return nil
}
