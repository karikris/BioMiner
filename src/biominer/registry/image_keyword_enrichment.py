"""Resumable, explicitly local name evidence for an image-bearing species cohort."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.registry.enrichment import SpeciesContext, STATIC_VERNACULAR_SOURCE_KEYS
from biominer.registry.query_eligibility import assess_name_query_eligibility

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "biominer-image-keyword-enrichment/v1"
SOURCE_ORDER = ("gbif_vernacular", "col", "wikidata", "inaturalist", "itis", "ala",
                "commons", "wikimedia", "open_tree", "tmd_de", "taxref_fr", "ncbi",
                *STATIC_VERNACULAR_SOURCE_KEYS, "mymemory")
SOURCE_LABELS = dict(zip(SOURCE_ORDER, (
    "GBIF names", "Catalogue of Life", "Wikidata", "iNaturalist", "ITIS",
    "Atlas of Living Australia", "Wikimedia Commons", "Wikipedia and Wikispecies",
    "Open Tree", "German TMD", "TAXREF France", "NCBI taxonomy",
    "Butterflies of India", "Bharat Ki Titliya", "Karnataka Chitte",
    "Papilio demoleus multilingual candidate table", "MyMemory dictionary"), strict=True))
MENTIONED_ONLY = ("eol", "bold", "libretranslate")
NAME_FIELDS = ("evidence_id", "accepted_taxon_key", "species_key", "scientific_name",
    "species_ordinal", "source_ordinal", "source_key", "source", "display_name", "language",
    "script", "region", "name_class", "source_record_id", "source_taxon_id", "lineage_check",
    "trust_tier", "confidence", "precision_tier", "enabled", "review_state", "disabled_reason",
    "query_eligible", "query_eligibility_reason", "retrieved_at", "source_response_hash",
    "source_url", "evidence_maturity", "source_assertion_json")
BOOL_FIELDS = {"enabled", "query_eligible"}
INT_FIELDS = {"species_ordinal", "source_ordinal"}
TERMINAL = {"complete", "unavailable", "input_conflict", "not_applicable"}


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name("." + path.name + "." + uuid4().hex)
    try:
        with tmp.open("x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def run_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("an enrichment coordinator already owns this output") from exc
        f.seek(0); f.truncate(); json.dump({"pid": os.getpid(), "started_at": now()}, f); f.flush()
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def code_inputs() -> dict[str, str]:
    paths = [Path(__file__), *[ROOT / "src/biominer/registry" / (x + ".py") for x in
        ("image_species", "image_name_sources", "image_keyword_http", "enrichment",
         "enrichment_sources", "translation_harvester", "translation_sources", "normalize",
         "query_eligibility", "trust_policy")], ROOT / "config/name_translation_target_locales.json",
         ROOT / "config/butterfly_scope.json", ROOT / "scripts/enrich_butterfly_image_species.py",
         ROOT / "scripts/start_butterfly_keyword_enrichment.sh"]
    paths += sorted((ROOT / "config/vernacular_sources").glob("*.json"))
    result = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    snapshots = [ROOT / "data/sources/ncbi/current" / n for n in ("ncbi_taxa.parquet", "ncbi_names.parquet")]
    snapshots += [ROOT / json.loads(p.read_bytes())["snapshot_path"]
                  for p in (ROOT / "config/vernacular_sources").glob("*.json")]
    for p in snapshots:
        result[str(p.relative_to(ROOT))] = sha(p) if p.is_file() else "missing"
    return result


class WorkState:
    def __init__(self, path: Path, config: dict):
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.executescript("""
        CREATE TABLE IF NOT EXISTS configuration (id INTEGER PRIMARY KEY, fingerprint TEXT, body TEXT);
        CREATE TABLE IF NOT EXISTS jobs (taxon TEXT, source TEXT, status TEXT, attempts INTEGER,
            name_count INTEGER, error_class TEXT, receipts TEXT, finished_at TEXT, outcome_reason TEXT, PRIMARY KEY(taxon,source));
        CREATE TABLE IF NOT EXISTS links (id TEXT PRIMARY KEY, taxon TEXT, source TEXT,
            source_taxon_id TEXT, body TEXT);
        CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, taxon TEXT, source TEXT, body TEXT);
        CREATE TABLE IF NOT EXISTS translation_units (id TEXT PRIMARY KEY, receipts TEXT,
            name_count INTEGER, finished_at TEXT, taxon TEXT, target_language TEXT, source_name TEXT);
        """)
        definitions = ",".join('"' + k + '" ' + ("INTEGER" if k in BOOL_FIELDS | INT_FIELDS else "TEXT")
                               for k in NAME_FIELDS)
        self.con.execute("CREATE TABLE IF NOT EXISTS names (" + definitions + ", PRIMARY KEY(evidence_id))")
        self.con.execute("CREATE INDEX IF NOT EXISTS names_taxon ON names(accepted_taxon_key)")
        fp = canonical_semantic_fingerprint(config)
        old = self.con.execute("SELECT fingerprint FROM configuration WHERE id=1").fetchone()
        if old and old[0] != fp:
            self.con.close()
            raise ValueError("resume input/source configuration changed; use a new output directory")
        if not old:
            self.con.execute("INSERT INTO configuration VALUES(1,?,?)", (fp, json.dumps(config, sort_keys=True)))
            self.con.commit()
        self.fingerprint = fp

    def status(self, taxon: str, source: str) -> str | None:
        row = self.con.execute("SELECT status FROM jobs WHERE taxon=? AND source=?", (taxon, source)).fetchone()
        return row[0] if row else None

    def names(self, taxon: str) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM names WHERE accepted_taxon_key=?", (taxon,))]

    def qitems(self, taxon: str) -> set[str]:
        return {r[0] for r in self.con.execute("SELECT source_taxon_id FROM links WHERE taxon=? AND lower(source)='wikidata'", (taxon,))}

    def _records(self, row: dict, key: str, ordinal: int, result: dict, receipts: list[dict]) -> None:
        response_hash = canonical_semantic_fingerprint([r.get("response_sha256", r.get("response_hash", "")) for r in receipts])
        if not receipts:
            response_hash = canonical_semantic_fingerprint(result.get("source_snapshots", []))
        for assertion in result.get("name_assertions", []):
            assertion = dict(assertion)
            if assertion.get("accepted_taxon_key") != row["accepted_taxon_key"]:
                raise ValueError("source assertion is bound to a different species")
            if assertion.get("source") == "Wikidata" and any(
                link.get("accepted_taxon_key") == row["accepted_taxon_key"]
                and link.get("source") == "Wikidata"
                and link.get("source_taxon_id") == assertion.get("source_taxon_id")
                and link.get("match_method") == "P225+P846+GBIF_COL_XR"
                and link.get("match_confidence") == "high"
                for link in result.get("external_links", [])):
                assertion["source_binding_confident"] = True
            item = {k: assertion.get(k, "") for k in NAME_FIELDS}
            item.update(accepted_taxon_key=row["accepted_taxon_key"], species_key=row["species_key"],
                scientific_name=row["scientific_name"], species_ordinal=row["species_ordinal"],
                source_ordinal=ordinal, source_key=key, retrieved_at=now(),
                source_response_hash=response_hash, evidence_maturity="source-bound name candidate; not verified taxonomy or image identity")
            item["source_assertion_json"] = json.dumps(assertion, ensure_ascii=False, sort_keys=True)
            if not item["source_url"]:
                item["source_url"] = next((str(s.get("source_path", s.get("source_url", "")))
                    for s in result.get("source_snapshots", []) if s.get("source_path", s.get("source_url"))), "")
            if not item["source_url"] and receipts:
                item["source_url"] = "https://" + receipts[0]["host"] + receipts[0]["path"]
            item["enabled"] = bool(assertion.get("enabled", False)) and bool(row.get("query_eligible", True))
            decision = assess_name_query_eligibility({**assertion, **item})
            item["query_eligible"] = bool(decision.query_eligible) and item["enabled"]
            item["query_eligibility_reason"] = decision.query_disabled_reason
            semantic = {k: v for k, v in item.items() if k not in {"evidence_id", "retrieved_at"}}
            item["evidence_id"] = canonical_semantic_fingerprint(semantic)
            self.con.execute("INSERT OR IGNORE INTO names VALUES(" + ",".join("?" for _ in NAME_FIELDS) + ")",
                             tuple(item[k] for k in NAME_FIELDS))
        for table, records in (("links", result.get("external_links", [])), ("snapshots", result.get("source_snapshots", []))):
            for record in records:
                if table == "links" and record.get("accepted_taxon_key") != row["accepted_taxon_key"]:
                    raise ValueError("source link is bound to a different species")
                identity = canonical_semantic_fingerprint([row["accepted_taxon_key"], key, record])
                if table == "links":
                    self.con.execute("INSERT OR IGNORE INTO links VALUES(?,?,?,?,?)", (identity, row["accepted_taxon_key"],
                        record.get("source", key), record.get("source_taxon_id", ""), json.dumps(record, sort_keys=True)))
                else:
                    self.con.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?)", (identity,
                        row["accepted_taxon_key"], key, json.dumps(record, sort_keys=True)))

    def commit(self, row: dict, key: str, ordinal: int, result: dict, *, status="complete", error="", receipts=()) -> None:
        with self.con:
            self._records(row, key, ordinal, result, list(receipts))
            self.con.execute("""INSERT INTO jobs VALUES(?,?,?,1,?,?,?,?,?) ON CONFLICT(taxon,source) DO UPDATE SET
                status=excluded.status,attempts=jobs.attempts+1,name_count=excluded.name_count,
                error_class=excluded.error_class,receipts=excluded.receipts,finished_at=excluded.finished_at,
                outcome_reason=excluded.outcome_reason""",
                (row["accepted_taxon_key"], key, status, len(result.get("name_assertions", [])), error,
                 json.dumps(list(receipts), sort_keys=True), now(), result.get("reason", "")))

    def translation_done(self, key: str) -> bool:
        return self.con.execute("SELECT 1 FROM translation_units WHERE id=?", (key,)).fetchone() is not None

    def translation_commit(self, unit: str, row: dict, ordinal: int, result: dict, receipts: list[dict],
                           *, target_language="", source_name="") -> None:
        with self.con:
            self._records(row, "mymemory", ordinal, result, receipts)
            self.con.execute("INSERT OR IGNORE INTO translation_units VALUES(?,?,?,?,?,?,?)", (unit,
                json.dumps(receipts, sort_keys=True), len(result.get("name_assertions", [])), now(),
                row["accepted_taxon_key"], target_language, source_name))

    def export(self, output: Path, metadata: dict) -> dict:
        target = output / "snapshots" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
        target.mkdir(parents=True)
        artifacts = []
        tables = ("names", "jobs", "links", "snapshots", "translation_units")
        for table in tables:
            cursor = self.con.execute('SELECT * FROM "' + table + '" ORDER BY 1')
            columns = [c[0] for c in cursor.description]
            schema = pa.schema([(c, pa.bool_() if table == "names" and c in BOOL_FIELDS else
                pa.int64() if (table == "names" and c in INT_FIELDS) or (table in {"jobs", "translation_units"} and c in {"attempts", "name_count"})
                else pa.string()) for c in columns])
            path = target / (table + ".parquet")
            count = 0
            with pq.ParquetWriter(path, schema, compression="zstd") as writer:
                while batch := cursor.fetchmany(1000):
                    rows = [dict(r) for r in batch]
                    if table == "names":
                        for r in rows:
                            for c in BOOL_FIELDS: r[c] = bool(r[c])
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema)); count += len(rows)
            artifacts.append({"path": path.name, "sha256": sha(path), "rows": count, "bytes": path.stat().st_size})
        counts = {r[0]: r[1] for r in self.con.execute("SELECT status,count(*) FROM jobs GROUP BY status")}
        manifest = {"schema": SCHEMA, "created_at": now(), "configuration_fingerprint": self.fingerprint,
                    **metadata, "source_status_counts": counts, "artifacts": artifacts, "manifest_written_last": True}
        atomic_json(target / "manifest.json", manifest)
        atomic_json(output / "latest.json", {"manifest": str((target / "manifest.json").relative_to(output)),
            "sha256": sha(target / "manifest.json"), "state": metadata["state"], "created_at": now()})
        return manifest


class SourceBundle:
    def __init__(self, http, state: WorkState, rows: list[dict], locales: tuple[str, ...], emit: Callable):
        from biominer.registry import enrichment_sources as e
        from biominer.registry import image_name_sources as n
        from biominer.registry.translation_harvester import WikimediaLanglinksProvider, MyMemoryTranslationProvider
        self.http, self.state, self.rows, self.locales, self.emit = http, state, rows, locales, emit
        self.qids: dict[str, set[str]] = {}
        self.clients = {
            "gbif_vernacular": n.GBIFImageNamesClient(http_get=http.get(n.GBIF_BASE_URL), resolve_legacy_by_name=True),
            "col": e.CatalogueOfLifeClient(http_get=http.get("https://api.checklistbank.org")),
            "wikidata": e.WikidataClient(http_get=http.get(e.WIKIDATA_WDQS_URL),
                col_xr_resolver=e.GBIFCOLXRLegacyResolver(http_get=http.get(e.GBIF_API_URL)),
                legacy_ids_require_crosswalk=True),
            "inaturalist": e.INaturalistClient(http_get=http.get("https://api.inaturalist.org")),
            "itis": e.ITISClient(http_get=http.get("https://www.itis.gov")),
            "ala": n.ALAImageNamesClient(http_get=http.get(n.ALA_BASE_URL)),
            "commons": n.CommonsImageNamesClient(http_get=http.get(n.COMMONS_BASE_URL), expected_wikidata_items=self.qids),
            "wikimedia": WikimediaLanglinksProvider(http_get=http.get("https://en.wikipedia.org"),
                wikispecies_http_get=http.get("https://species.wikimedia.org")),
            "open_tree": e.OpenTreeBulkClient(json_post=http.post("https://api.opentreeoflife.org/v3")),
            "tmd_de": e.TMDGermanClient(graphql_post=lambda payload: http.graphql(e.TMD_TAXONOMY_GRAPHQL_URL)(payload["query"], payload.get("variables", {}))),
            "taxref_fr": e.TAXREFFrenchClient(http_get=http.get(e.TAXREF_BASE_URL)),
            "mymemory": MyMemoryTranslationProvider(http_get=http.get("https://api.mymemory.translated.net"),
                email=os.environ.get("MYMEMORY_EMAIL"), api_key=os.environ.get("MYMEMORY_API_KEY"), allow_machine_translation=False),
        }
        self.unavailable = {}
        ncbi = ROOT / "data/sources/ncbi/current"
        if all((ncbi / name).is_file() for name in ("ncbi_taxa.parquet", "ncbi_names.parquet")):
            self.clients["ncbi"] = e.NCBIBulkClient(source_dir=ncbi)
        else: self.unavailable["ncbi"] = "required_current_taxonomy_snapshots_missing"
        for key in STATIC_VERNACULAR_SOURCE_KEYS:
            path = ROOT / "config/vernacular_sources" / (key + ".json")
            config = json.loads(path.read_bytes())
            snapshot = ROOT / config["snapshot_path"]
            if not snapshot.is_file(): self.unavailable[key] = "configured_snapshot_missing"
            else: self.clients[key] = e.StaticVernacularSourceClient.from_config_path(path, snapshot_root=ROOT / "data/source_snapshots")
        self.bulk: dict[str, dict] = {}
        self.bulk_receipts: dict[str, list[dict]] = {}
        self.taxa = [{"accepted_taxon_key": r["accepted_taxon_key"], "scientific_name": r["scientific_name"],
            "family": r["family"], "genus": r["genus"], "rank": "SPECIES"} for r in rows if r.get("query_eligible", True)]

    def run(self, key: str, context: SpeciesContext, row: dict, ordinal: int) -> dict:
        from biominer.registry.enrichment_sources import _name_assertion
        from biominer.registry.translation_harvester import _wikimedia_binding_decision
        self.qids[context.accepted_taxon_key] = self.state.qitems(context.accepted_taxon_key)
        client = self.clients[key]
        if key == "wikimedia":
            assertions, snapshots = [], []
            for method in (client.langlinks, client.vernacular_names):
                links, _, _ = method(context.accepted_scientific_name, target_locales=self.locales)
                for link in links:
                    reason, match, lineage = _wikimedia_binding_decision(link, self.qids[context.accepted_taxon_key])
                    assertion = _name_assertion(context, link.title, source="Wikimedia",
                        source_record_id=f"{link.page_id}:{link.language}:{link.title}", source_taxon_id=link.wikidata_item,
                        language=link.language, script="", trust_tier="T3", enabled=not reason,
                        review_state="candidate" if reason else "accepted", disabled_reason=reason,
                        lineage_check="accepted_taxon_key" if not reason else "")
                    assertion.update(observed_source_taxon_id=link.wikidata_item,
                        expected_source_taxon_ids=sorted(self.qids[context.accepted_taxon_key]),
                        source_binding_status=match, source_binding_evidence=lineage)
                    assertions.append(assertion)
            return {"name_assertions": assertions, "source_snapshots": snapshots}
        if key == "mymemory":
            seeds = sorted({r["display_name"] for r in self.state.names(context.accepted_taxon_key)
                if r["language"] in {"en", "eng"} and r["query_eligible"] and r["name_class"] in {"vernacular", "vernacular_alias", "common_name"}})
            if not seeds: return {"work_status": "not_applicable", "reason": "no_eligible_english_common_name"}
            for seed in seeds[:1]:
                for locale in self.locales:
                    if locale in {"en", "eng"}: continue
                    unit = canonical_semantic_fingerprint([context.accepted_taxon_key, seed, locale])
                    if self.state.translation_done(unit): continue
                    self.http.reset_job_receipts()
                    values, _, _ = client.translate(source_name=seed, source_language="en", target_language=locale, max_candidates=3)
                    names = [_name_assertion(context, name, source="MyMemory", source_record_id=unit + ":" + name,
                        language=locale, script="", name_class="generated_translation", trust_tier="T5",
                        enabled=False, review_state="candidate", disabled_reason="dictionary_translation_requires_review") for name in values]
                    self.state.translation_commit(unit, row, ordinal, {"name_assertions": names}, self.http.job_receipts,
                        target_language=locale, source_name=seed)
                    self.emit(event="translation_unit", target_language=locale, status="complete", names=len(names))
            return {"name_assertions": []}
        if hasattr(client, "enrich_species"): return client.enrich_species(context)
        if key == "taxref_fr":
            return client.enrich_registry(taxa_rows=[r for r in self.taxa if r["accepted_taxon_key"] == context.accepted_taxon_key], name_rows=[])
        if key not in self.bulk:
            self.emit(event="bulk_source", status="started")
            result = client.enrich_registry(taxa_rows=self.taxa, name_rows=[])
            self.bulk[key] = result
            self.bulk_receipts[key] = list(self.http.job_receipts)
            self.emit(event="bulk_source", status="complete", names=len(result.get("name_assertions", [])))
        else:
            self.http.job_receipts.extend(self.bulk_receipts[key])
        result = self.bulk[key]
        return {"name_assertions": [a for a in result.get("name_assertions", []) if a["accepted_taxon_key"] == context.accepted_taxon_key],
            "external_links": [a for a in result.get("external_links", []) if a["accepted_taxon_key"] == context.accepted_taxon_key],
            "source_snapshots": result.get("source_snapshots", [])}


def enrich_inventory(collection: Path, output: Path, *, handlers: dict[str, Callable] | None = None,
                     sources=SOURCE_ORDER, event_sink: Callable | None = None, **options) -> dict:
    with run_lock(output):
        return _enrich_locked(collection, output, handlers=handlers, sources=tuple(sources), event_sink=event_sink, **options)


def _enrich_locked(collection: Path, output: Path, *, handlers=None, sources=SOURCE_ORDER,
        event_sink=None, checkpoint_every=250, max_species=0, max_seconds=86400,
        max_requests=250000, min_interval=1.0, max_retries=3, locales=None) -> dict:
    from biominer.registry.image_keyword_http import KeywordHTTP, BudgetDeferred, KeywordHTTPError
    if checkpoint_every < 1 or max_species < 0 or max_seconds <= 0:
        raise ValueError("invalid run bounds")
    if len(sources) != len(set(sources)) or not sources or any(s not in SOURCE_ORDER for s in sources):
        raise ValueError("sources must be unique known source keys")
    manifest = json.loads((collection / "manifest.json").read_bytes())
    for item in manifest["artifacts"]:
        if sha(collection / item["path"]) != item["sha256"]: raise ValueError("species collection artifact changed")
    rows = pq.read_table(collection / "species.parquet").to_pylist()
    if len(rows) != manifest["species_count"]: raise ValueError("species count differs from collection")
    if not rows: raise ValueError("species collection is empty; inspect source-format and exclusion diagnostics")
    variants: dict[str, list[str]] = {}
    for row in pq.read_table(collection / "source_name_variants.parquet").to_pylist():
        variants.setdefault(row["accepted_taxon_key"], []).append(row["scientific_name"])
    locales = tuple(locales or json.loads((ROOT / "config/name_translation_target_locales.json").read_bytes()))
    inputs = code_inputs()
    config = {"schema": SCHEMA, "collection_manifest_sha256": sha(collection / "manifest.json"),
        "collection_fingerprint": manifest["semantic_fingerprint"], "sources": sources, "locales": locales,
        "code_inputs": inputs, "local_mode": True, "machine_translation": False, "english_seeds_per_species": 1}
    state = WorkState(output / "work.sqlite3", config)
    http = None
    cursor = {"species_ordinal": 0, "species_total": len(rows), "source_ordinal": 0,
              "source_total": len(sources), "source_key": "", "scientific_name": ""}
    log = logging.getLogger("biominer.image_keywords")
    handler = logging.FileHandler(output / "progress.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")); log.addHandler(handler); log.setLevel(logging.INFO)
    for name in ("httpx", "httpcore"): logging.getLogger(name).setLevel(logging.WARNING)

    def emit(event="progress", **fields):
        record = {**cursor, "event": event, **fields, "updated_at": now()}
        log.info("%d/%d species | %d/%d sources | source=%s | species=%s | %s",
            cursor["species_ordinal"], len(rows), cursor["source_ordinal"], len(sources),
            cursor["source_key"], cursor["scientific_name"], json.dumps(fields, ensure_ascii=False, sort_keys=True))
        atomic_json(output / "status.json", record)
        if event_sink: event_sink(record)

    started = time.monotonic(); run_state = "running"
    metadata = {"species_total": len(rows), "source_total": len(sources), "sources": list(sources),
        "source_labels": SOURCE_LABELS, "unavailable_sources": {}, "mentioned_only_without_clients": list(MENTIONED_ONLY),
        "local_mode": True, "runtime_registered": False, "production_migration_performed": False,
        "collection_manifest": {"path": str((collection / "manifest.json").resolve()), "sha256": sha(collection / "manifest.json")},
        "producer_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    try:
        http = KeywordHTTP(output / "http.sqlite3", max_retries=max_retries, min_interval=min_interval,
                           max_requests=max_requests, logger=lambda event: emit(**event)) if handlers is None else None
        bundle = SourceBundle(http, state, rows, locales, emit) if handlers is None else None
        unavailable = bundle.unavailable if bundle else {}
        metadata["unavailable_sources"] = unavailable
        blocked_sources = {}
        emit(event="run_started", status="running", unavailable_sources=unavailable)
        for row in rows[:max_species or None]:
            cursor.update(species_ordinal=row["species_ordinal"], scientific_name=row["scientific_name"])
            context = SpeciesContext(row["accepted_taxon_key"], row["scientific_name"], "gbif:" + row["family_key"],
                row["family"], "", row["genus"], tuple(variants.get(row["accepted_taxon_key"], [row["scientific_name"]])))
            for ordinal, key in enumerate(sources, 1):
                cursor.update(source_ordinal=ordinal, source_key=key)
                if time.monotonic() - started >= max_seconds:
                    run_state = "paused_runtime_limit"; return {"state": run_state, **cursor}
                previous = state.status(context.accepted_taxon_key, key)
                if previous in TERMINAL:
                    emit(status="resume_cached", previous_status=previous); continue
                status, error, result = "complete", "", {}
                if http: http.reset_job_receipts()
                if not row.get("query_eligible", True): status = "input_conflict"
                elif key in unavailable: status = "unavailable"
                elif key in blocked_sources:
                    status, result = "deferred", {"reason": blocked_sources[key]}
                else:
                    emit(status="started")
                    try:
                        if http:
                            with http.source(key): result = bundle.run(key, context, row, ordinal)
                        else: result = handlers[key](context)
                        status = result.get("work_status", "complete")
                    except BudgetDeferred as exc:
                        status, error = "deferred", type(exc).__name__
                        result = {"reason": str(exc)}
                    except KeywordHTTPError as exc:
                        status, error = "error", type(exc).__name__
                        result = {"reason": str(exc)}
                        if str(exc) in {"http_401", "http_403"}:
                            blocked_sources[key] = "provider_authorization_failure_" + str(exc)
                    except Exception as exc:
                        status, error = "error", type(exc).__name__
                if status == "unavailable": result = {**result, "reason": unavailable.get(key, result.get("reason", "source_unavailable"))}
                if status == "input_conflict": result = {"reason": row.get("binding_status", "input_conflict")}
                state.commit(row, key, ordinal, result, status=status, error=error,
                             receipts=http.job_receipts if http else ())
                emit(status=status, error_class=error, reason=result.get("reason", ""), names=len(result.get("name_assertions", [])))
            if row["species_ordinal"] == 1 or row["species_ordinal"] % checkpoint_every == 0:
                if code_inputs() != inputs: raise RuntimeError("source/configuration changed during enrichment")
                state.export(output, {**metadata, "state": "running", **cursor})
        incomplete = state.con.execute("SELECT count(*) FROM jobs WHERE status IN ('error','deferred','unavailable','input_conflict')").fetchone()[0]
        run_state = "completed_with_gaps" if incomplete else ("partial_species_limit" if max_species and max_species < len(rows) else "complete")
        return {"state": run_state, **cursor}
    except BaseException:
        run_state = "interrupted"; raise
    finally:
        try:
            final = state.export(output, {**metadata, "state": run_state, **cursor})
            emit(event="run_checkpoint", status=run_state, source_status_counts=final["source_status_counts"])
        finally:
            state.con.close()
            if http: http.close()
            log.removeHandler(handler); handler.close()
