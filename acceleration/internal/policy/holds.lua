-- One subject's holds, shared by every router: a hash whose fields are usage ids and whose
-- values are "<micros>:<expires unix ms>:<router>".
--
-- KEYS[1] is the hash. ARGV[1] is now in unix ms and ARGV[2] the calling router's id; then
-- pairs of field and value: a value puts the field, and an empty one drops it if this router
-- put it, so a router letting go of a hold never drops one another router put since.
--
-- It drops the expired fields, applies the changes, keeps the hash until its last field
-- expires, and returns what the other routers hold and how many fields are live.
local now = tonumber(ARGV[1])
local me = ARGV[2]
for i = 3, #ARGV, 2 do
	local field, value = ARGV[i], ARGV[i + 1]
	if value == '' then
		local current = redis.call('HGET', KEYS[1], field)
		if current and string.match(current, ':([^:]*)$') == me then
			redis.call('HDEL', KEYS[1], field)
		end
	else
		redis.call('HSET', KEYS[1], field, value)
	end
end
local all = redis.call('HGETALL', KEYS[1])
local others, live, latest = 0, 0, 0
for i = 1, #all, 2 do
	local micros, expires, owner = string.match(all[i + 1], '^(-?%d+):(%d+):(.*)$')
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
