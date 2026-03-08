# Postflop Per-Street Solve Cache

## Overview

Cache solved postflop spots to disk so repeated visits skip the solve. Each street (flop/turn/river) is cached independently, keyed by a hash of the full config + board + action path that led to it.

## Cache Location

Inside the blueprint directory: `{blueprint_dir}/spots/{hash}.bin`

The blueprint directory is not yet wired up in the postflop explorer — that's a prerequisite.

## Cache Key

Hash of all inputs that determine the solution (excluding exploitability):

- **Flop**: `hash(ranges + pot + stack + bet_sizes + flop_cards)`
- **Turn**: `hash(ranges + pot + stack + bet_sizes + flop_cards + flop_action_history + turn_card)`
- **River**: `hash(ranges + pot + stack + bet_sizes + flop_cards + flop_actions + turn_card + turn_actions + river_card)`

For turn/river, `ranges` refers to the original config ranges (not filtered weights), since the filtered weights are deterministic given the same flop solution + action history.

Use a deterministic hash (e.g., SHA-256 truncated to 16 bytes, or xxhash64) over the canonicalized inputs.

## Cache Value (File Format)

```
[header]
  magic: u32          = 0x50534343 ("PSCC" - PostflopSolveCache)
  version: u32        = 1
  exploitability: f32  (achieved, for display only)
  iterations: u32      (how many iterations were run)
  storage1_len: u64
  storage2_len: u64
  storage_ip_len: u64
  storage_chance_len: u64

[body]
  storage1: [u8; storage1_len]
  storage2: [u8; storage2_len]
  storage_ip: [u8; storage_ip_len]
  storage_chance: [u8; storage_chance_len]
```

The config itself is NOT stored in the file — it's implicit from the cache key. On load, we rebuild the game from config, allocate memory, then overwrite the storage buffers.

## Load Flow

1. User picks board (or closes a street to advance)
2. Compute cache key from current config + board + action history
3. Check `{blueprint_dir}/spots/{key_hex}.bin`
4. If found:
   - Rebuild game from config via `build_game()` + `allocate_memory(false)`
   - Read storage buffers from file, overwrite game's `storage1/2/ip/chance`
   - Set game state to `Solved`
   - Call `finalize()` (or cache post-finalize buffers to skip this)
   - Store in `state.game`, set `solve_complete = true`
   - Frontend sees solved state immediately
5. If not found:
   - Show solve button as today

## Save Flow

1. `postflop_solve_street_core` completes (solve thread finishes)
2. Compute cache key (same as load)
3. Write header + storage buffers to `{blueprint_dir}/spots/{key_hex}.bin`
4. Save is fire-and-forget (don't block the UI)

## UX Considerations

- When a cached solve loads, show metadata: "Cached (exploitability: X% pot, N iterations)"
- User can choose to re-solve (overwrites cache)
- No manual cache management needed initially (can add cache clear later)

## Open Questions

- Should `finalize()` output be cached (so we skip finalize on load), or should we re-finalize from raw buffers? Caching post-finalize is faster but may store more data.
- Exact hash function choice (xxhash64 is fast, SHA-256 is collision-proof)

## Prerequisites

- Blueprint directory must be wired up in the postflop explorer (separate task)
