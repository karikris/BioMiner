# Frontend dependency findings — report only

Date: 2026-10-01. Scope: TaxaLens and ButterflyLens `apps/web`; no installs, upgrades, repository writes, commits, pushes, or compatibility-pin changes. Both existing dirty worktrees were preserved. No application builds, unit/browser tests, exploit tests, or hosted checks ran.

## Exact inventory and checks

| Repository | Inspected HEAD | Lock SHA-256 |
|---|---|---|
| TaxaLens | `03374097e172e117fa643a0bfebd75e727e3ce5c` | `921b2cc0d409f01abfec64dfd97a4a205925198167dad7630fdfc22c15910b81` |
| ButterflyLens | `72eb2f656bf78def0d9ca759623b7d1e7ef62cb7` | `975d095422f4c6909ebe79f2212b8c58f006491fef63d1810d4a6c0c403c32ac` |

Lock hashes matched at the end. Relevant installed `node_modules` versions matched their lock entries. Environment: Node `22.22.1`, npm `9.2.0`; manifests allow Node `>=22.12`, npm `>=9`.

Both pin Vite `8.1.4`, Vitest `4.1.10`, plugin-react `6.0.3`, jsdom `29.1.1`, TypeScript `7.0.2`, React/ReactDOM `19.2.7`. Rolldown is `1.1.5`. Neither lock contains esbuild, Rollup, Next, or ESLint, so historical findings against those packages are not assigned to these current trees.

Read-only commands in each `apps/web` directory:

- `npm audit --package-lock-only --json --cache /tmp/biominer-dependencies-20261001-k3kxy0ga/npm-cache`
- `npm audit --omit=dev --package-lock-only --json --cache /tmp/biominer-dependencies-20261001-k3kxy0ga/npm-cache`

All-dependency audits exited 1 because findings are present. TaxaLens: 7 vulnerable package entries (1 critical, 3 high, 3 moderate), representing 27 unique advisory URLs. ButterflyLens: 5 vulnerable package entries (2 high, 3 moderate), representing 18 unique URLs. Vitest and its mocker duplicate the same advisory in the package counts. Production-only audit: TaxaLens 2 package entries (`fast-uri`, `maplibre-gl`); ButterflyLens 0. These are dependency/advisory classifications, not verified exploitability or a deployment security certification.

## Shared tooling findings and compatible repair candidates

| Installed chain, both frontends | Security issue | Fix candidate |
|---|---|---|
| `vitest 4.1.10 → @vitest/mocker 4.1.10` | Path traversal/arbitrary file read via redirect-mock WebSocket registration; [GHSA-82fw-gwwq-j7x9](https://github.com/advisories/GHSA-82fw-gwwq-j7x9), CVE-2026-84373, CVSS 5.9 | Pin Vitest `4.1.11` and regenerate the aligned `@vitest/* 4.1.11` lock entries. GitHits patch review reports no peer changes. |
| `vite 8.1.4 → postcss 8.5.19` | Source-map file read with attacker-controlled annotation when `from` is missing; [GHSA-fxqj-rqcc-2cmp](https://github.com/advisories/GHSA-fxqj-rqcc-2cmp), CVE-2026-69153 | Refresh PostCSS to `8.5.28` (minimum fix `8.5.23`; `8.5.26` also improves symlink protection). Fits Vite's `^8.5.16`. |
| `vite → postcss → nanoid 3.3.16` | Zero-size custom-generator infinite loop; [GHSA-2v37-7h3g-55p8](https://github.com/advisories/GHSA-2v37-7h3g-55p8), CVE-2026-67213 | Refresh Nanoid to `3.3.18`, within PostCSS's existing 3.x range; PostCSS `8.5.28` raises its lower bound to `^3.3.18`. |
| `jsdom 29.1.1 → undici 7.28.0` | HTTP cache disclosure, retry framing, cookie/blob injection, TLS callback loss in BalancedPool, WebSocket/decompression failures; see full advisory table below | Refresh Undici to `7.29.1` or a reviewed later 7.x, within jsdom's `^7.25.0`. This addresses the listed current audit ranges without taking jsdom 30. |

Dev/test exposure inference: both ordinary dev and preview scripts bind `127.0.0.1`. Configured Vitest tests use jsdom; no browser/API test server or explicit mockerPlugin integration was found in the inspected configs. The vulnerable mocker WebSocket route is a developer/test server surface, and these tooling packages are dev-only lock entries. Loopback binding reduces remote reach but does not establish immunity to local or browser-origin attacks. No running-server or exploit check was performed.

GitHits source comparison (release notes were absent for Vitest `4.1.11`): `npm:@vitest/mocker@4.1.10`, `src/node/interceptorPlugin.ts` joins the untrusted redirect path then registers it without a file check. Version `4.1.11`, lines 79–88, adds `isFileLoadingAllowed(server.config, redirect)` before registration and can disable raw WebSocket events using `registerWebSocketEvents:false`. `load` subsequently reads registered redirect files. This supports the fix and server scope without inventing release-note coverage.

PostCSS source: `npm:postcss@8.5.23`, `lib/previous-map.js:87–95` rejects untrusted map loads without a CSS source filename and files outside its directory. Vite `8.1.4`, `src/node/plugins/css.ts:1735–1739`, passes `from:source` explicitly. The advisory's missing-from trigger is therefore not evident in Vite's normal CSS call; the installed vulnerable package should still be refreshed. Locally inspected PostCSS uses `nanoid/non-secure` with size `6`, rather than a zero-size custom generator; no direct application custom-generator use was found.

## TaxaLens production-library findings

- `maplibre-gl 5.24.0` is affected by critical [GHSA-jrc7-96c5-q579](https://github.com/advisories/GHSA-jrc7-96c5-q579), CVE-2026-85061, CVSS 10. Fixed from `6.4.1`; GitHits reports no patched 5.x range. Its critical advisory classification must stay separate from current application reachability. `OfflineWorldMap.tsx` disables attribution control and renders React Popup children; GitHits inspected `@vis.gl/react-maplibre 8.1.1`, `src/components/popup.ts:52`, uses `setDOMContent`, not HTML string sanitization. Current MapLibre source uses `DOM.sanitize` for attribution HTML. These facts reduce the obvious local trigger; they do not clear the affected dependency.
- `ajv 8.20.0 → fast-uri 3.1.3` is affected by eight direct GitHits advisories (seven high, one medium) involving host/authority parsing and normalization. Refresh to `3.1.8`, which satisfies Ajv's `^3.0.1` and clears all eight in the direct GitHits review. These are production lock entries. Local Ajv use is strict synchronous schema validation; no `loadSchema` or `compileAsync` call was found, so a direct application SSRF path was not established.

MapLibre `5.24.0 → 6.11.2` was reviewed in a GitHits batch; the target clears the sanitizer advisory but requires a separate migration. The [v5→v6 guide](https://github.com/maplibre/maplibre-gl-js/blob/acb7b722f6a752ac1a2ba98cf83006c83dc4fafa/docs/guides/v5-to-v6-migration-guide.md) establishes ESM-only distribution, no default export/CJS/UMD/CSP bundle, a bundler worker URL requirement, event class/type changes, native nested GeoJSON properties, missing-image resolver changes, and WebGL2-only support. TaxaLens's current `OfflineWorldMap.tsx:316` accepts WebGL1 as supported, and would need updating with fallback checks before this upgrade. The React wrapper's peer is `maplibre-gl >=4.0.0`, and its loader accepts named `Map` exports, but permissive peer metadata is not tested 6.x compatibility. Map scripts/notices contain the exact `5.24.0` string and must be updated with the browser, worker/offline-network, fallback, visual, map-distribution, and licence checks. No sibling change was made.

## Other reviewed versions and compatibility boundaries

GitHits current latest versions: Vite `8.3.1`, Vitest `5.0.3`, jsdom `30.1.1`, PostCSS `8.5.28`, Nanoid `6.0.1`, Undici `8.11.2`, fast-uri `4.2.1`, MapLibre `6.11.2`.

Vite `8.1.4` has no active direct advisories. `8.1.4 → 8.3.1` adds build/dev-server features and updates Rolldown from the 1.1 range to 1.2, CSS/PostCSS/picomatch ranges, optimizer, HMR, worker and sourcemap behavior. This optional minor upgrade should get the normal unit/build/worker/offline/visual checks if adopted. Updating Vite alone cannot be assumed to repair every independently frozen lock entry.

Vitest `5.0.3` also fixes the mocker issue, but is a major version with rewritten dependency structure and a higher Vite floor (`^6.4.0 || ^7.0.0 || ^8.0.0`). GitHits did not have release-note bodies for this range; do not recommend an untested major solely for this fix when `4.1.11` addresses it.

jsdom `30.1.1` requires Node `^22.22.2 || ^24.15.0 || >=26.0.0`, as explicitly documented in its 30.0 release notes. Current Node `22.22.1` fails that floor, and both manifests' `>=22.12` is too permissive even for installed jsdom 29 (`^20.19.0 || ^22.13.0 || >=24.0.0`). jsdom 30 also changes DOM/CSS/focus behavior and moves to Undici 8. A same-line Undici lock refresh is the narrower security repair. Future jsdom 30 work must first align Node engine/CI and run DOM/a11y/focus tests.

## Evidence disagreements and coverage limits

- GitHits package upgrade transitive summaries resolve registry dependency ranges; they do not inspect these local locks. They reported zero transitive vulnerabilities for Vite/jsdom, whereas pinned individual checks and npm exact-lock audits expose the retained older transitive versions. Decisions here use exact locks and pinned checks.
- `pkg_vulns` and upgrade review returned nine Undici advisories. npm audit returned fifteen affecting `7.28.0`; the [upstream 7.29.1 release](https://github.com/nodejs/undici/releases/tag/v7.29.1), retrieved through GitHits, independently documents the six additional advisories: `GHSA-w293-vg96-wgc3`, `GHSA-rx4f-c7p8-82vq`, `GHSA-3xpg-4rpp-hhhm`, `GHSA-8436-99hf-9mmv`, `GHSA-2gqq-gqf2-x968`, `GHSA-r53p-7pc4-xj5r`. All npm affected ranges end at `7.29.1`.
- The upstream fast-uri `3.1.7` release additionally lists `GHSA-58mr-gqgx-xq4g`, not returned by either the pinned GitHits vulnerability response or npm audit. Its affected range was not independently established here; it is a source coverage gap, not included in the 27/18 counts.
- npm assigns Nanoid high severity, while GitHits gives medium/CVSS5.9. npm PostCSS CVSS is zero/unspecified; GitHits reports CVSS6. Preserve source classifications rather than invent one reconciled score.
- Some source queries returned interim indexed refs. Pinned mocker/Vite/PostCSS and wrapper reads ultimately produced focused code. Undici7.29.1 source indexing was pending for the attempted engine grep; compatibility advice relies on exact parent range, direct upgrade review and upstream release history, not a claimed full source diff. No application tests ran.

## GitHits provenance

`quick_start` loaded and its untrusted-content rules applied. Used `pkg_info` for eight packages listed above; `pkg_vulns` pinned Vite8.1.4, Vitest4.1.10, PostCSS8.5.19, Nanoid3.3.16, Undici7.28.0, MapLibre5.24.0 and fast-uri3.1.3; batched `pkg_upgrade_review` (3 candidates, then 2 narrow candidates, then 5 security candidates); explicit `pkg_changelog` for Vitest4.1.10..4.1.11, Vite8.1.4..8.3.1, jsdom29.1.1..30.1.1, Undici7.28.0..7.29.1, fast-uri3.1.3..3.1.8, PostCSS8.5.19..8.5.28.

Source queries included `redirect mock path traversal root startsWith browser middleware`, `sourceMappingURL source map from customRandom size zero`, `sanitize attributes setDOMContent setHTML Popup peer maplibre`, and deterministic greps for `redirect`, `redirectUrl`, `isFileLoadingAllowed`, `postcss(`, `opts.from`, `sanitize(`, `module.default`, zero-size branches, and Undici's Node engine. Search continuations were followed with `search_status` refs `nVfo4ZLgz70gcbMwqT-fpg`, `Qb6P_fqvuutg2N5sxA8T8A`, and `HRlNZW7PwUuPBHLRtntXFA`. Negative grep results, pagination and indexing limitations were not treated as full-source absence.

Repositories reviewed: `vitejs/vite`, `vitest-dev/vitest`, `postcss/postcss`, `ai/nanoid`, `jsdom/jsdom`, `nodejs/undici`, `fastify/fast-uri`, `maplibre/maplibre-gl-js`, `visgl/react-map-gl`. Vite/Vitest/PostCSS/Nanoid/jsdom/Undici/React wrapper are MIT; fast-uri/MapLibre are BSD-3-Clause. No external code was copied. GitHits package/release/source evidence was used instead of sending sibling application source to the public service. Provenance was not appended to sibling repositories because this scope is report only.

## Exact npm advisory records

The tables below deduplicate direct npm advisory rows by URL. Affected ranges are the npm registry audit's package ranges, not independently proven application triggers. Full raw audits are temporary JSON files under `/tmp/biominer-dependencies-20261001-k3kxy0ga`; they are not committed alongside this report.

### taxalens

| Package | Advisory | npm severity | npm affected range |
|---|---|---|---|
| `undici` | [GHSA-2gqq-gqf2-x968](https://github.com/advisories/GHSA-2gqq-gqf2-x968) | low | `>=7.1.0 <7.29.1` |
| `undici` | [GHSA-2jfj-6hjv-fm6j](https://github.com/advisories/GHSA-2jfj-6hjv-fm6j) | moderate | `>=7.0.0 <7.29.1` |
| `nanoid` | [GHSA-2v37-7h3g-55p8](https://github.com/advisories/GHSA-2v37-7h3g-55p8) | high | `<3.3.18` |
| `undici` | [GHSA-3wwx-pv8p-q78v](https://github.com/advisories/GHSA-3wwx-pv8p-q78v) | moderate | `>=7.28.0 <7.29.1` |
| `undici` | [GHSA-3xpg-4rpp-hhhm](https://github.com/advisories/GHSA-3xpg-4rpp-hhhm) | moderate | `>=7.15.0 <7.29.1` |
| `undici` | [GHSA-4cwx-7wf7-3272](https://github.com/advisories/GHSA-4cwx-7wf7-3272) | high | `>=7.0.0 <7.29.0` |
| `fast-uri` | [GHSA-5jgf-p345-68v8](https://github.com/advisories/GHSA-5jgf-p345-68v8) | high | `>=3.1.3 <3.1.6` |
| `fast-uri` | [GHSA-7p8r-x3mc-p8w7](https://github.com/advisories/GHSA-7p8r-x3mc-p8w7) | high | `>=3.0.0 <3.1.5` |
| `@vitest/mocker` | [GHSA-82fw-gwwq-j7x9](https://github.com/advisories/GHSA-82fw-gwwq-j7x9) | moderate | `>=2.1.0 <4.1.11` |
| `undici` | [GHSA-8436-99hf-9mmv](https://github.com/advisories/GHSA-8436-99hf-9mmv) | low | `>=7.0.0 <7.29.1` |
| `undici` | [GHSA-8xcm-r25x-g524](https://github.com/advisories/GHSA-8xcm-r25x-g524) | moderate | `>=7.0.0 <7.29.0` |
| `fast-uri` | [GHSA-f65p-4m7j-42xc](https://github.com/advisories/GHSA-f65p-4m7j-42xc) | high | `>=3.0.0 <3.1.6` |
| `fast-uri` | [GHSA-fph4-wmhf-6fwf](https://github.com/advisories/GHSA-fph4-wmhf-6fwf) | high | `>=3.1.2 <3.1.6` |
| `postcss` | [GHSA-fxqj-rqcc-2cmp](https://github.com/advisories/GHSA-fxqj-rqcc-2cmp) | moderate | `<=8.5.22` |
| `fast-uri` | [GHSA-hrr3-gc8f-f4qj](https://github.com/advisories/GHSA-hrr3-gc8f-f4qj) | moderate | `>=3.0.0 <3.1.8` |
| `fast-uri` | [GHSA-jqff-g426-hqxp](https://github.com/advisories/GHSA-jqff-g426-hqxp) | high | `>=3.0.0 <3.1.6` |
| `undici` | [GHSA-jr45-8vmc-qm54](https://github.com/advisories/GHSA-jr45-8vmc-qm54) | moderate | `>=7.0.0 <7.29.0` |
| `maplibre-gl` | [GHSA-jrc7-96c5-q579](https://github.com/advisories/GHSA-jrc7-96c5-q579) | critical | `<=6.4.0` |
| `undici` | [GHSA-m8rv-5g2x-5cg5](https://github.com/advisories/GHSA-m8rv-5g2x-5cg5) | moderate | `>=7.0.0 <7.29.0` |
| `undici` | [GHSA-pmjh-fq2x-6v4x](https://github.com/advisories/GHSA-pmjh-fq2x-6v4x) | moderate | `>=7.11.0 <7.29.1` |
| `fast-uri` | [GHSA-qw65-cvwx-89v3](https://github.com/advisories/GHSA-qw65-cvwx-89v3) | high | `>=3.0.0 <3.1.7` |
| `undici` | [GHSA-r53p-7pc4-xj5r](https://github.com/advisories/GHSA-r53p-7pc4-xj5r) | low | `>=7.0.0 <7.29.1` |
| `undici` | [GHSA-rfgv-xxqx-mfg5](https://github.com/advisories/GHSA-rfgv-xxqx-mfg5) | high | `>=7.0.0 <7.29.1` |
| `undici` | [GHSA-rx4f-c7p8-82vq](https://github.com/advisories/GHSA-rx4f-c7p8-82vq) | moderate | `>=7.0.0 <7.29.1` |
| `fast-uri` | [GHSA-v2hh-gcrm-f6hx](https://github.com/advisories/GHSA-v2hh-gcrm-f6hx) | high | `>=3.0.0 <=3.1.3` |
| `undici` | [GHSA-v3r7-h72x-cjcm](https://github.com/advisories/GHSA-v3r7-h72x-cjcm) | moderate | `>=7.0.0 <7.29.0` |
| `undici` | [GHSA-w293-vg96-wgc3](https://github.com/advisories/GHSA-w293-vg96-wgc3) | high | `>=7.24.1 <7.29.1` |

### butterflylens

| Package | Advisory | npm severity | npm affected range |
|---|---|---|---|
| `undici` | [GHSA-2gqq-gqf2-x968](https://github.com/advisories/GHSA-2gqq-gqf2-x968) | low | `>=7.1.0 <7.29.1` |
| `undici` | [GHSA-2jfj-6hjv-fm6j](https://github.com/advisories/GHSA-2jfj-6hjv-fm6j) | moderate | `>=7.0.0 <7.29.1` |
| `nanoid` | [GHSA-2v37-7h3g-55p8](https://github.com/advisories/GHSA-2v37-7h3g-55p8) | high | `<3.3.18` |
| `undici` | [GHSA-3wwx-pv8p-q78v](https://github.com/advisories/GHSA-3wwx-pv8p-q78v) | moderate | `>=7.28.0 <7.29.1` |
| `undici` | [GHSA-3xpg-4rpp-hhhm](https://github.com/advisories/GHSA-3xpg-4rpp-hhhm) | moderate | `>=7.15.0 <7.29.1` |
| `undici` | [GHSA-4cwx-7wf7-3272](https://github.com/advisories/GHSA-4cwx-7wf7-3272) | high | `>=7.0.0 <7.29.0` |
| `@vitest/mocker` | [GHSA-82fw-gwwq-j7x9](https://github.com/advisories/GHSA-82fw-gwwq-j7x9) | moderate | `>=2.1.0 <4.1.11` |
| `undici` | [GHSA-8436-99hf-9mmv](https://github.com/advisories/GHSA-8436-99hf-9mmv) | low | `>=7.0.0 <7.29.1` |
| `undici` | [GHSA-8xcm-r25x-g524](https://github.com/advisories/GHSA-8xcm-r25x-g524) | moderate | `>=7.0.0 <7.29.0` |
| `postcss` | [GHSA-fxqj-rqcc-2cmp](https://github.com/advisories/GHSA-fxqj-rqcc-2cmp) | moderate | `<=8.5.22` |
| `undici` | [GHSA-jr45-8vmc-qm54](https://github.com/advisories/GHSA-jr45-8vmc-qm54) | moderate | `>=7.0.0 <7.29.0` |
| `undici` | [GHSA-m8rv-5g2x-5cg5](https://github.com/advisories/GHSA-m8rv-5g2x-5cg5) | moderate | `>=7.0.0 <7.29.0` |
| `undici` | [GHSA-pmjh-fq2x-6v4x](https://github.com/advisories/GHSA-pmjh-fq2x-6v4x) | moderate | `>=7.11.0 <7.29.1` |
| `undici` | [GHSA-r53p-7pc4-xj5r](https://github.com/advisories/GHSA-r53p-7pc4-xj5r) | low | `>=7.0.0 <7.29.1` |
| `undici` | [GHSA-rfgv-xxqx-mfg5](https://github.com/advisories/GHSA-rfgv-xxqx-mfg5) | high | `>=7.0.0 <7.29.1` |
| `undici` | [GHSA-rx4f-c7p8-82vq](https://github.com/advisories/GHSA-rx4f-c7p8-82vq) | moderate | `>=7.0.0 <7.29.1` |
| `undici` | [GHSA-v3r7-h72x-cjcm](https://github.com/advisories/GHSA-v3r7-h72x-cjcm) | moderate | `>=7.0.0 <7.29.0` |
| `undici` | [GHSA-w293-vg96-wgc3](https://github.com/advisories/GHSA-w293-vg96-wgc3) | high | `>=7.24.1 <7.29.1` |
