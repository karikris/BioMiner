from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


REQUIRED_DWC_FIELDS = [
    "occurrenceID",
    "basisOfRecord",
    "eventDate",
    "scientificName",
    "verbatimIdentification",
    "identificationVerificationStatus",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "verbatimLocality",
    "georeferenceSources",
    "georeferenceRemarks",
    "associatedMedia",
    "associatedReferences",
    "license",
    "rightsHolder",
    "dataGeneralizations",
    "informationWithheld",
    "occurrenceRemarks",
    "dynamicProperties",
]


def map_candidate_to_dwc(candidate: dict[str, Any]) -> dict[str, Any]:
    scientific_name = candidate["resolved_scientific_name"]
    event_date = candidate["eventDate"]
    latitude = candidate.get("decimalLatitude")
    longitude = candidate.get("decimalLongitude")
    occurrence_id = sha256(
        f"flickr{candidate['flickr_photo_id']}{scientific_name}{event_date}{latitude}{longitude}".encode("utf-8")
    ).hexdigest()
    dynamic_properties = dict(candidate.get("dynamicProperties", {}))
    evidence_provenance = _evidence_provenance(candidate)
    if evidence_provenance:
        dynamic_properties["evidence_provenance"] = evidence_provenance
    row = {
        "occurrenceID": occurrence_id,
        "basisOfRecord": "HumanObservation" if candidate.get("human_evidence") else "MachineObservation",
        "eventDate": event_date,
        "scientificName": scientific_name,
        "verbatimIdentification": candidate.get("verbatimIdentification"),
        "identificationVerificationStatus": candidate.get("identificationVerificationStatus", "needs_review"),
        "decimalLatitude": latitude,
        "decimalLongitude": longitude,
        "coordinateUncertaintyInMeters": candidate.get("coordinateUncertaintyInMeters"),
        "verbatimLocality": candidate.get("verbatimLocality"),
        "georeferenceSources": candidate.get("georeferenceSources"),
        "georeferenceRemarks": candidate.get("georeferenceRemarks"),
        "associatedMedia": candidate.get("associatedMedia"),
        "associatedReferences": candidate.get("associatedReferences"),
        "license": candidate.get("license"),
        "rightsHolder": candidate.get("rightsHolder"),
        "dataGeneralizations": candidate.get("dataGeneralizations"),
        "informationWithheld": candidate.get("informationWithheld"),
        "occurrenceRemarks": candidate.get("occurrenceRemarks"),
        "dynamicProperties": json.dumps(dynamic_properties, sort_keys=True),
    }
    return row


def _evidence_provenance(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "publication_state",
        "publication_state_reason",
        "review_reason",
        "species_text_match",
        "human_verification_detected",
        "human_verification_terms",
        "bioclip_top1_label",
        "bioclip_top1_score",
        "bioclip_species_agreement_status",
    )
    return {key: candidate[key] for key in keys if key in candidate}
