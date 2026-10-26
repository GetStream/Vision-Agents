-- One subject's holds, shared by every router: a hash whose fields are "<usage id>:<router>",
-- so no router ever writes over another's, and whose values are "<micros>:<expires unix ms>",
-- the expiry by Redis's own clock.
--
-- KEYS[1] is the hash and ARGV[1] the calling router's id; then pairs of usage id and value:
-- "<micros>:<ttl ms>" puts this router's field for that id, to expire ttl from now, and an
-- empty value drops it.
--
-- It drops the expired fields, applies the changes, keeps the hash until its last field
-- expires, and returns what the other routers hold and how many fields are live.
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local me = ARGV[1]
for i = 2, #ARGV, 2 do
	local field, value = ARGV[i] .. ':' .. me, ARGV[i + 1]
	if value == '' then
		redis.call('HDEL', KEYS[1], field)
	else
		local micros, ttl = string.match(value, '^(-?%d+):(%d+)$')
		redis.call('HSET', KEYS[1], field, micros .. ':' .. (now + tonumber(ttl)))
	end
end
local all = redis.call('HGETALL', KEYS[1])
local others, live, latest = 0, 0, 0
for i = 1, #all, 2 do
	local owner = string.match(all[i], ':([^:]*)$')
	local micros, expires = string.match(all[i + 1], '^(-?%d+):(%d+)$')
	expires = tonumber(expires)
	if not expires or expires <= now then
		redis.call('HDEL', KEYS[1], all[i])
	else
		live = live + 1
		if expires > latest then
			latest = expires
		end
		if owner ~= me then
			others = others + tonumber(micros)
		end
	end
end
if latest > 0 then
	redis.call('PEXPIREAT', KEYS[1], latest)
end
return { others, live }
