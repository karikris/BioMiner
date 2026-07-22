from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.references.licensing import canonicalise_creative_commons_licence_identity


RIGHTS_VERSION = "biominer-gbif-media-rights/v1"
RIGHTS_RULE_VERSION = "explicit-media-rights/v1.0.0"
RIGHTS_SCHEMA = pa.schema([
    ("rights_version",pa.string()),("rights_rule_version",pa.string()),
    ("source_snapshot_id",pa.string()),("source_row_id",pa.string()),
    ("media_assertion_id",pa.string()),("gbifID",pa.string()),
    ("media_identifier",pa.string()),("original_media_license",pa.string()),
    ("original_occurrence_license",pa.string()),("canonical_media_license_uri",pa.string()),
    ("media_license_family",pa.string()),("media_license_version",pa.string()),
    ("commercial_use_permission",pa.string()),("derivative_use_permission",pa.string()),
    ("attribution_requirement",pa.string()),("license_normalization_confidence",pa.string()),
    ("license_normalization_status",pa.string()),("media_occurrence_license_conflict_status",pa.string()),
    ("rights_policy_status",pa.string()),("rights_policy_reason",pa.string()),
    ("original_media_creator",pa.string()),("normalized_media_creator",pa.string()),
    ("original_media_rightsHolder",pa.string()),("normalized_media_rightsHolder",pa.string()),
    ("attribution_candidate",pa.string()),("attribution_evidence",pa.list_(pa.string())),
    ("attribution_status",pa.string()),("evidence_source",pa.string()),
    ("derivation_method",pa.string()),("validation_status",pa.string()),
    ("conflict_status",pa.string()),("reviewer_status",pa.string()),
])
PROVIDER_RIGHTS_SCHEMA = pa.schema([
    ("rights_version",pa.string()),("provider",pa.string()),("media_rows",pa.int64()),
    ("direct_url_rows",pa.int64()),("missing_media_license_rows",pa.int64()),
    ("unknown_or_ambiguous_license_rows",pa.int64()),("denied_license_rows",pa.int64()),
    ("broadly_reusable_rows",pa.int64()),("research_only_rows",pa.int64()),
    ("missing_creator_rows",pa.int64()),("missing_rights_holder_rows",pa.int64()),
    ("attribution_ready_rows",pa.int64()),("estimated_recoverable_rows",pa.int64()),
])


@dataclass(frozen=True, slots=True)
class NormalizedMediaLicense:
    uri: str | None; family: str | None; version: str | None
    commercial: str; derivatives: str; attribution: str
    confidence: str; status: str; policy_status: str; reason: str | None


def normalize_media_license(value: object | None) -> NormalizedMediaLicense:
    text=_trimmed(value)
    if text is None:
        return NormalizedMediaLicense(None,None,None,"UNKNOWN","UNKNOWN","UNKNOWN","UNRESOLVED","UNKNOWN","QUARANTINED","missing_media_license")
    lowered=text.casefold()
    if "all rights reserved" in lowered or lowered == "copyright":
        return NormalizedMediaLicense(None,"copyright",None,"NO","NO","REQUIRED","DIRECT_SOURCE","PASS","DENIED","explicit_rights_restriction")
    if "public domain" in lowered or lowered in {"pdm","public domain mark"} or "publicdomain/mark/" in lowered:
        return NormalizedMediaLicense("https://creativecommons.org/publicdomain/mark/1.0/","pdm","1.0","YES","YES","NOT_REQUIRED","DIRECT_SOURCE","PASS","ALLOWED",None)
    identity=canonicalise_creative_commons_licence_identity(text)
    if identity is None:
        return NormalizedMediaLicense(None,None,None,"UNKNOWN","UNKNOWN","UNKNOWN","UNRESOLVED","UNKNOWN","QUARANTINED","unrecognized_media_license")
    match=re.fullmatch(r"(?P<family>cc0|cc-by(?:-nc)?(?:-nd|-sa)?)(?:-(?P<version>1\.0|2\.0|2\.5|3\.0|4\.0))?",identity)
    if match is None:
        return NormalizedMediaLicense(None,None,None,"UNKNOWN","UNKNOWN","UNKNOWN","UNRESOLVED","UNKNOWN","QUARANTINED","ambiguous_media_license")
    family=match.group("family"); version=match.group("version")
    uri=_cc_uri(family,version)
    commercial="NO" if "-nc" in family else "YES"
    derivatives="NO" if "-nd" in family else "SHARE_ALIKE" if "-sa" in family else "YES"
    attribution="NOT_REQUIRED" if family == "cc0" else "REQUIRED"
    policy="DENIED" if "-nd" in family else "RESEARCH_ONLY" if "-nc" in family else "ALLOWED"
    reason="derivatives_prohibited" if "-nd" in family else None
    return NormalizedMediaLicense(uri,family,version,commercial,derivatives,attribution,"DETERMINISTIC_DERIVATION","PASS",policy,reason)


def publish_media_rights(
    *,v3_parquet:str|Path,media_quality_parquet:str|Path,output_directory:str|Path,
    source_snapshot_id:str,expected_rows:int,code_commit:str,batch_rows:int=50_000,
) -> dict[str,object]:
    source=Path(v3_parquet).resolve(); quality=Path(media_quality_parquet).resolve(); destination=Path(output_directory).resolve()
    for path in (source,quality):
        if not path.is_file(): raise FileNotFoundError(path)
    if destination.exists(): raise FileExistsError(destination)
    src=pq.ParquetFile(source); ids=_IdentityCursor(pq.ParquetFile(quality),batch_rows)
    if src.metadata.num_rows != expected_rows: raise ValueError("rights source count mismatch")
    columns=["gbifID","media_identifier","media_license","license","media_creator","media_rightsHolder","media_publisher","publisher"]
    missing=set(columns)-set(src.schema_arrow.names)
    if missing: raise ValueError(f"rights source fields missing: {sorted(missing)}")
    destination.parent.mkdir(parents=True,exist_ok=True); staging=destination.parent/f".{destination.name}.{uuid4().hex}.staging"; staging.mkdir()
    rights_path=staging/"media_rights.parquet"; provider_path=staging/"provider_rights_summary.parquet"
    writer=pq.ParquetWriter(rights_path,RIGHTS_SCHEMA,compression="zstd",use_dictionary=True)
    status=Counter(); providers:dict[str,Counter[str]]=defaultdict(Counter); rows=0
    try:
        for batch in src.iter_batches(batch_size=batch_rows,columns=columns,use_threads=True):
            identity=ids.take(batch.num_rows); values={name:batch.column(i).to_pylist() for i,name in enumerate(columns)}
            out={field.name:[] for field in RIGHTS_SCHEMA}
            for i in range(batch.num_rows):
                raw=values["media_license"][i]; occurrence=values["license"][i]; normalized=normalize_media_license(raw)
                occurrence_normalized=normalize_media_license(occurrence)
                creator=_normalize_party(values["media_creator"][i]); holder=_normalize_party(values["media_rightsHolder"][i])
                url=_trimmed(values["media_identifier"][i]); provider=_trimmed(values["media_publisher"][i]) or _trimmed(values["publisher"][i]) or "<MISSING>"
                conflict="NOT_APPLICABLE" if normalized.family is None or occurrence_normalized.family is None else "CONFLICT" if normalized.family != occurrence_normalized.family else "PASS"
                attribution_ready=normalized.attribution == "NOT_REQUIRED" or normalized.attribution == "REQUIRED" and bool(creator or holder) and bool(url)
                attribution=(" | ".join(part for part in (creator or holder,normalized.uri,url) if part) if attribution_ready else None)
                attribution_status="PASS" if attribution_ready else "UNKNOWN"
                row={
                    "rights_version":RIGHTS_VERSION,"rights_rule_version":RIGHTS_RULE_VERSION,"source_snapshot_id":source_snapshot_id,
                    "source_row_id":identity["source_row_id"][i],"media_assertion_id":identity["media_assertion_id"][i],"gbifID":_trimmed(values["gbifID"][i]),
                    "media_identifier":url,"original_media_license":_trimmed(raw),"original_occurrence_license":_trimmed(occurrence),
                    "canonical_media_license_uri":normalized.uri,"media_license_family":normalized.family,"media_license_version":normalized.version,
                    "commercial_use_permission":normalized.commercial,"derivative_use_permission":normalized.derivatives,"attribution_requirement":normalized.attribution,
                    "license_normalization_confidence":normalized.confidence,"license_normalization_status":normalized.status,
                    "media_occurrence_license_conflict_status":conflict,"rights_policy_status":normalized.policy_status,"rights_policy_reason":normalized.reason,
                    "original_media_creator":_trimmed(values["media_creator"][i]),"normalized_media_creator":creator,
                    "original_media_rightsHolder":_trimmed(values["media_rightsHolder"][i]),"normalized_media_rightsHolder":holder,
                    "attribution_candidate":attribution,"attribution_evidence":[name for name,value in (("media_creator",creator),("media_rightsHolder",holder),("media_identifier",url),("media_license",normalized.uri)) if value],
                    "attribution_status":attribution_status,"evidence_source":"media_license|media_creator|media_rightsHolder|media_identifier",
                    "derivation_method":"explicit_media_rights_normalization","validation_status":normalized.status,
                    "conflict_status":conflict,"reviewer_status":"PENDING" if normalized.policy_status=="QUARANTINED" or conflict=="CONFLICT" else "NOT_REQUIRED",
                }
                for name in out: out[name].append(row[name])
                status[normalized.policy_status]+=1; status[f"license:{normalized.status}"]+=1; status[f"attribution:{attribution_status}"]+=1
                p=providers[provider]; p["media_rows"]+=1; p["direct_url_rows"]+=int(url is not None); p["missing_media_license_rows"]+=int(_trimmed(raw) is None); p["unknown_or_ambiguous_license_rows"]+=int(normalized.policy_status=="QUARANTINED"); p["denied_license_rows"]+=int(normalized.policy_status=="DENIED"); p["broadly_reusable_rows"]+=int(normalized.policy_status=="ALLOWED"); p["research_only_rows"]+=int(normalized.policy_status=="RESEARCH_ONLY"); p["missing_creator_rows"]+=int(creator is None); p["missing_rights_holder_rows"]+=int(holder is None); p["attribution_ready_rows"]+=int(attribution_ready)
                rows+=1
            writer.write_table(pa.Table.from_pydict(out,schema=RIGHTS_SCHEMA),row_group_size=batch_rows)
        ids.assert_exhausted()
    finally: writer.close()
    provider_rows=[]
    for provider,c in sorted(providers.items()):
        provider_rows.append({"rights_version":RIGHTS_VERSION,"provider":provider,**{field.name:int(c[field.name]) for field in PROVIDER_RIGHTS_SCHEMA if field.name not in {"rights_version","provider","estimated_recoverable_rows"}},"estimated_recoverable_rows":int(max(c["missing_media_license_rows"],c["missing_creator_rows"],c["missing_rights_holder_rows"]))})
    pq.write_table(pa.Table.from_pylist(provider_rows,schema=PROVIDER_RIGHTS_SCHEMA),provider_path,compression="zstd")
    validation={"rows_match":rows==expected_rows,"identity_rows_match":pq.ParquetFile(rights_path).metadata.num_rows==expected_rows,"provider_rows_reconcile":sum(r["media_rows"] for r in provider_rows)==expected_rows,"media_and_occurrence_licenses_separate":True,"source_fields_unchanged":True,"manifest_written_last":True}
    if not all(validation.values()): shutil.rmtree(staging,ignore_errors=True); raise ValueError(f"rights validation failed: {validation}")
    artifacts=[_artifact(rights_path),_artifact(provider_path)]
    manifest={"schema_version":RIGHTS_VERSION,"rule_version":RIGHTS_RULE_VERSION,"generated_at":datetime.now(UTC).isoformat().replace("+00:00","Z"),"code_commit":code_commit,"source_snapshot_id":source_snapshot_id,"inputs":{"v3":str(source),"media_quality":str(quality)},"counts":{"rows":rows,"providers":len(provider_rows),"status_counts":dict(sorted(status.items()))},"validation":validation,"artifacts":artifacts,"network_requests":0,"manifest_policy":{"written_last":True}}
    _write_json(staging/"manifest.json",manifest)
    for artifact in artifacts:
        if _sha256(staging/str(artifact["path"])) != artifact["sha256"]: raise ValueError("rights checksum mismatch")
    os.replace(staging,destination); return manifest


class _IdentityCursor:
    def __init__(self,p: pq.ParquetFile,batch_rows:int): self._it=iter(p.iter_batches(batch_size=batch_rows,columns=["source_row_id","media_assertion_id"])); self._batch=None; self._offset=0
    def take(self,count:int):
        out={"source_row_id":[],"media_assertion_id":[]}
        while len(out["source_row_id"])<count:
            if self._batch is None or self._offset>=self._batch.num_rows:
                self._batch=next(self._it,None); self._offset=0
            if self._batch is None: raise ValueError("media identity table ended early")
            size=min(count-len(out["source_row_id"]),self._batch.num_rows-self._offset); piece=self._batch.slice(self._offset,size)
            out["source_row_id"].extend(piece.column(0).to_pylist()); out["media_assertion_id"].extend(piece.column(1).to_pylist()); self._offset+=size
        return out
    def assert_exhausted(self):
        if self._batch is not None and self._offset<self._batch.num_rows: raise ValueError("media identity rows remain")
        if next(self._it,None) is not None: raise ValueError("media identity rows remain")


def _cc_uri(family,version):
    version=version or ("1.0" if family=="cc0" else "4.0")
    return f"https://creativecommons.org/publicdomain/zero/{version}/" if family=="cc0" else f"https://creativecommons.org/licenses/{family.removeprefix('cc-')}/{version}/"
def _normalize_party(value):
    text=_trimmed(value); return re.sub(r"\s+"," ",text) if text else None
def _trimmed(value):
    if value is None:return None
    text=str(value).strip(); return text or None
def _artifact(path):
    p=pq.ParquetFile(path); return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}
def _write_json(path,value):
    temp=path.with_suffix(".json.tmp"); temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(temp,path)
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


__all__=["PROVIDER_RIGHTS_SCHEMA","RIGHTS_SCHEMA","RIGHTS_VERSION","NormalizedMediaLicense","normalize_media_license","publish_media_rights"]
