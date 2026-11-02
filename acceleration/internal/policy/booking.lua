-- One organization's bookings, counted by every router: a hash whose fields are booking ids and
-- whose values are "<micros>:<expires unix ms>:<charged, 0 or 1>", the expiry by Redis's own
-- clock, so a router that died lets go of its bookings once they expire.
--
-- KEYS[1] is the hash. ARGV[1] is the booking id and ARGV[2] what to do with it:
--   "book", with ARGV[3] its micros, ARGV[4] its ttl in ms, ARGV[5] the room the credit leaves
--   before any booking and ARGV[6] the least the booking needs of that room: it is added only
--   when the room less every other live booking covers that;
--   "charge", with ARGV[3] a ttl in ms, which marks it charged and has it expire by then;
--   "release", which drops it.
--
-- It drops the expired bookings first, and returns whether the booking is held, what the other
-- live bookings add up to, and how many of those it dropped had never been charged.
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local id, verb = ARGV[1], ARGV[2]
local all = redis.call('HGETALL', KEYS[1])
local others, expired, latest = 0, 0, 0
local mine
for i = 1, #all, 2 do
	local micros, expires, charged = string.match(all[i + 1], '^(-?%d+):(%d+):([01])$')
	expires = tonumber(expires)
	if not expires or expires <= now then
		redis.call('HDEL', KEYS[1], all[i])
		if charged == '0' then
			expired = expired + 1
		end
	elseif all[i] == id then
		mine = {micros, expires}
	else
		others = others + tonumber(micros)
		if expires > latest then
			latest = expires
		end
	end
end
local held = 0
if verb == 'book' then
	local ttl, room, need = tonumber(ARGV[4]), tonumber(ARGV[5]), tonumber(ARGV[6])
	if room - others >= need then
		redis.call('HSET', KEYS[1], id, ARGV[3] .. ':' .. (now + ttl) .. ':0')
		held = 1
		latest = math.max(latest, now + ttl)
	end
elseif verb == 'charge' and mine then
	local expires = math.min(mine[2], now + tonumber(ARGV[3]))
	redis.call('HSET', KEYS[1], id, mine[1] .. ':' .. expires .. ':1')
	held = 1
	latest = math.max(latest, expires)
elseif verb == 'release' and mine then
	redis.call('HDEL', KEYS[1], id)
end
if latest > 0 then
	redis.call('PEXPIREAT', KEYS[1], latest)
end
return {held, others, expired}
