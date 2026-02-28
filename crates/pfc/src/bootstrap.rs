use rusty_genius_core::error::GeniusError;

/// Detected Redis capabilities for choosing search strategy.
#[derive(Debug, Clone)]
pub struct RedisCapabilities {
    pub has_redisearch: bool,
}

/// Lua script for brute-force cosine similarity over `pfc:vec:*` keys.
/// KEYS[1] = prefix (e.g. "pfc")
/// ARGV[1] = JSON-encoded query vector
/// ARGV[2] = limit
/// Returns alternating [id, score, id, score, ...]
pub const LUA_COSINE_SEARCH: &str = r#"
local cjson = require('cjson')
local prefix = KEYS[1]
local query = cjson.decode(ARGV[1])
local limit = tonumber(ARGV[2])
local keys = redis.call('KEYS', prefix .. ':vec:*')
local results = {}
for _, key in ipairs(keys) do
  local raw = redis.call('GET', key)
  if raw then
    local vec = cjson.decode(raw)
    local dot, na, nb = 0, 0, 0
    for i = 1, #query do
      dot = dot + query[i] * vec[i]
      na = na + query[i] * query[i]
      nb = nb + vec[i] * vec[i]
    end
    local sim = 0
    if na > 0 and nb > 0 then sim = dot / (math.sqrt(na) * math.sqrt(nb)) end
    local id = string.sub(key, #prefix + 6)
    results[#results+1] = {sim, id}
  end
end
table.sort(results, function(a, b) return a[1] > b[1] end)
local ret = {}
for i = 1, math.min(limit, #results) do
  ret[#ret+1] = results[i][2]; ret[#ret+1] = tostring(results[i][1])
end
return ret
"#;

/// Detect Redis capabilities (RediSearch availability).
pub async fn detect_capabilities(
    conn: &mut redis::aio::MultiplexedConnection,
) -> RedisCapabilities {
    let has_redisearch = redis::cmd("FT._LIST")
        .query_async::<Vec<String>>(conn)
        .await
        .is_ok();
    RedisCapabilities { has_redisearch }
}

/// Create FTS index on `pfc:obj:*` hashes if RediSearch is available.
pub async fn create_redisearch_index(
    conn: &mut redis::aio::MultiplexedConnection,
) -> Result<(), GeniusError> {
    // Check if index already exists
    let exists: Result<redis::Value, _> = redis::cmd("FT.INFO")
        .arg("pfc:idx")
        .query_async(conn)
        .await;
    if exists.is_ok() {
        return Ok(());
    }

    redis::cmd("FT.CREATE")
        .arg("pfc:idx")
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg("1")
        .arg("pfc:obj:")
        .arg("SCHEMA")
        .arg("short_name")
        .arg("TEXT")
        .arg("long_name")
        .arg("TEXT")
        .arg("description")
        .arg("TEXT")
        .arg("content")
        .arg("TEXT")
        .arg("object_type")
        .arg("TAG")
        .query_async::<redis::Value>(conn)
        .await
        .map_err(|e| GeniusError::MemoryError(format!("FT.CREATE failed: {}", e)))?;

    Ok(())
}
