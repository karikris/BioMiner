# Refactor MCP Environment

Recorded for branch `cleanup/production-workflow-postgres-s3` during Phase 0.

## Available MCP / App Tools

| Tool/server | Status | Evidence | Use in Phase 0 |
| --- | --- | --- | --- |
| GitHub app MCP | Available | `mcp__codex_apps__github._get_repo` returned metadata for `karikris/BioMiner` with push/admin permissions. | Used for repository access verification. |
| Morph MCP | Tool metadata exposed, but unavailable for local search | `mcp__morph_mcp.codebase_search` returned HTTP 429. | Marked unavailable for this goal; will not be called again unless the goal is restarted. |
| GitHits MCP | Available in tool discovery | `mcp__githits` tools exposed by `tool_search`. | Available for later external examples/research if needed; not useful for this local Phase 0 audit. |
| Gmail app MCP | Available in tool discovery | Gmail tools exposed by `tool_search`. | Not relevant to BioMiner refactor work. |

## Unavailable / Not Exposed

| Tool/server | Status | Impact |
| --- | --- | --- |
| Valyu MCP | Not exposed by tool discovery | No current external documentation lookup through Valyu in Phase 0. Use web/official docs only if later phases require current external API verification. |
| Headroom MCP | Not exposed by tool discovery | No session compression/retrieval support through Headroom in Phase 0. |

## Notes

- Repository has moved to sibling layout under `/Users/merm0001/Repos`: `BioMiner`, `YOLO26`, `BioCLIP25`, and `secrets`.
- Secrets loader currently resolves `/Users/merm0001/Repos/secrets/secrets.env` through sibling-base fallback.
- Real secret values were not printed; only presence/length checks were used in earlier verification.

