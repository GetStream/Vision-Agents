package main

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// startEpisodeSweeper starts the closer's idle sweeper only where an episode can be: with
// connectors on, whose channel bridge opens a thread's episodes, or with at least one agent
// config that has episode_cards on, whose calls open theirs. With neither it never starts, so
// nothing is swept, written or asked of Stream (Kanat, 2026-10-07, D3 of wave 3b). It reports
// whether it started. The check is made once, at start: a config that turns episode_cards on
// later has its calls' episodes closed by the call hook, and its expired summary leases taken
// again from the next start.
func startEpisodeSweeper(ctx context.Context, settings config.Config, pgStore *store.Store, closer *omnichannel.Closer) (bool, error) {
	if !settings.Connectors.Enabled {
		carded, err := pgStore.AnyEpisodeCards(ctx)
		if err != nil || !carded {
			return false, err
		}
	}
	closer.Start()
	return true, nil
}
