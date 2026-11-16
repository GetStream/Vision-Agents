-- Writes window totals into the subject hashes of one organization.
--
-- KEYS are the hashes. ARGV is six values per total: the index of its hash in KEYS, the
-- field ("<interval>:<YYYYMMDD>"), the total in micros, the interval, the oldest field of that
-- interval to keep (the previous window's), and when the hash may expire (unix seconds).
--
-- A field only ever goes up: a late or repeated write, or two replicas racing, end on the
-- larger total. A missing field is written even at zero, so a subject that never spent has
-- its field and is not read through again.
--
-- The contract for readers: the current window's field is the total. A reader reads through
-- the service (POST /v1/windows, which writes the field) when that field is missing, not only
-- when the key is: a key written in an earlier window holds that window's field alone.
--
-- Runs on Redis 5 and later, and on Valkey: no EXPIRETIME, which is 7.0's.

-- greater compares two non-negative integers written in decimal, exactly: a Lua number is a
-- double, which is not past 2^53.
local function greater(a, b)
  if #a ~= #b then
    return #a > #b
  end
  return a > b
end

local expiry, keeps = {}, {}
for i = 1, #ARGV, 6 do
  local key = KEYS[tonumber(ARGV[i])]
  local field, total, interval, keep, at = ARGV[i + 1], ARGV[i + 2], ARGV[i + 3], ARGV[i + 4], tonumber(ARGV[i + 5])

  local current = redis.call('HGET', key, field)
  if not current or greater(total, current) then
    redis.call('HSET', key, field, total)
  end

  keeps[key] = keeps[key] or {}
  if (keeps[key][interval] or '') < keep then
    keeps[key][interval] = keep
  end
  if (expiry[key] or 0) < at then
    expiry[key] = at
  end
end

-- One HKEYS a hash: the fields of each interval before the newest one to keep go.
for key, oldest in pairs(keeps) do
  for _, name in ipairs(redis.call('HKEYS', key)) do
    local interval = string.match(name, '^([^:]+):')
    local keep = interval and oldest[interval]
    if keep and name < keep then
      redis.call('HDEL', key, name)
    end
  end
end

local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
for key, at in pairs(expiry) do
  local ttl = redis.call('PTTL', key)
  if ttl < 0 or now + ttl < at * 1000 then
    redis.call('EXPIREAT', key, at)
  end
end
return 0
