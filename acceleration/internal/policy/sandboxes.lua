-- One organization's running sandboxes, counted by every router: a sorted set of slot ids,
-- each scored by when its lease lapses by Redis's own clock, so a router that died gives its
-- slots back once their leases lapse.
--
-- KEYS[1] is the set. ARGV[1] is the slot id and ARGV[2] its lease in ms. ARGV[3] is "take",
-- with ARGV[4] the most that may run at once and ARGV[5] "1" to take it whatever the count;
-- "renew", which keeps a slot held, putting it back if its lease lapsed while its sandbox ran
-- on; or "release", which gives it back.
--
-- It drops the lapsed slots, then does as asked, and returns whether the slot is held and how
-- many others are.
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local id, lease, verb = ARGV[1], tonumber(ARGV[2]), ARGV[3]
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if verb == 'release' then
	redis.call('ZREM', KEYS[1], id)
	return {0, redis.call('ZCARD', KEYS[1])}
end
if verb == 'take' then
	local others = redis.call('ZCARD', KEYS[1])
	if others >= tonumber(ARGV[4]) and ARGV[5] ~= '1' then
		return {0, others}
	end
end
redis.call('ZADD', KEYS[1], now + lease, id)
local latest = redis.call('ZRANGE', KEYS[1], -1, -1, 'WITHSCORES')
redis.call('PEXPIREAT', KEYS[1], tonumber(latest[2]))
return {1, redis.call('ZCARD', KEYS[1]) - 1}
