# Papilio demoleus Language Discovery

This guide documents the Papilio demoleus multilingual registry configuration. It is a regional name-discovery aid, not a Darwin Core publishing claim and not taxonomic validation.

## Inputs

| Artifact | Role |
|---|---|
| `config/range_seed/papilio_demoleus.json` | Seeds known/cautionary regions and country labels for GBIF occurrence-country discovery |
| `config/language_targets/papilio_demoleus_region_language_targets.json` | Maps configured regions to priority languages |
| `config/vernacular_sources/boi_india_en.json` | Curated static source config for Butterflies of India English names |
| `config/vernacular_sources/bharat_ki_titliya_hi.json` | Curated static source config for Hindi names; current snapshot has no accepted source-name rows |
| `config/vernacular_sources/karnataka_chitte_kn.json` | Curated static source config for Kannada names; current snapshot has no accepted source-name rows |
| `data/source_snapshots/*/2026-07-static-v1.csv` | Versioned static CSV snapshots |

Run the curated-only build with:

```bash
uv run biominer registry build \
  --output-dir data/registry/papilio_demoleus_curated \
  --registry-version 2026-07-papilio-demoleus-curated-v1 \
  --scope-json config/butterfly_scope.json \
  --enrichment-sources col,inaturalist,itis,tmd_de,wikidata,gbif_vernacular,taxref_fr,boi_india_en,bharat_ki_titliya_hi,karnataka_chitte_kn \
  --range-discovery-source gbif \
  --range-seed-json config/range_seed/papilio_demoleus.json \
  --language-targets-json config/language_targets/papilio_demoleus_region_language_targets.json \
  --curated-static-source-config-dir config/vernacular_sources \
  --curated-static-source-snapshot-dir data/source_snapshots \
  --skip-translations
```

Add low-trust translation candidates with:

```bash
uv run biominer registry build \
  --output-dir data/registry/papilio_demoleus_multilingual \
  --registry-version 2026-07-papilio-demoleus-multilingual-v1 \
  --scope-json config/butterfly_scope.json \
  --enrichment-sources col,inaturalist,itis,tmd_de,wikidata,gbif_vernacular,taxref_fr,boi_india_en,bharat_ki_titliya_hi,karnataka_chitte_kn \
  --translation-sources wikimedia,mymemory \
  --translation-target-locales-json config/name_translation_target_locales.json \
  --range-discovery-source gbif \
  --range-seed-json config/range_seed/papilio_demoleus.json \
  --language-targets-json config/language_targets/papilio_demoleus_region_language_targets.json \
  --max-translation-candidates-per-name 3
```

## Regional Language Buckets

The countries/territories below are configured seed regions. GBIF occurrence discovery still determines which rows are emitted for a given build.

| Region | Countries/territories discovered | Priority languages | Curated sources available | Missing curated sources | Query-eligible names | Disabled/cautionary names |
|---|---|---|---|---|---|---|
| South Asia | Bangladesh, Bhutan, India, Nepal, Pakistan, Sri Lanka | English, Hindi, Bengali, Urdu, Punjabi, Nepali, Sinhala, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Odia, Assamese | Butterflies of India English; Bharat Ki Titliya Hindi configured but no rows; Karnataka Chitte Kannada configured but Karnataka-scoped | Bengali, Urdu, Punjabi, Nepali, Sinhala, Tamil, Telugu, Malayalam, Marathi, Gujarati, Odia, Assamese; Hindi/Kannada need accepted source rows | Lime Butterfly, Lime Swallowtail, Northern Lime Swallowtail from Butterflies of India when mapped cleanly | Ambiguous short/common terms such as Lime stay disabled |
| Middle East / West Asia | Bahrain, Iran, Iraq, Kuwait, Oman, Qatar, Saudi Arabia, Syria, Turkey, United Arab Emirates | Arabic, Persian, Kurdish, Turkish, English | None in current static snapshots | Arabic, Persian, Kurdish, Turkish, regional English source | None from current curated static sources | Generated/dictionary translations remain T5 and query-disabled by default |
| East Asia | China, Hong Kong, Japan, Macau, Taiwan | Simplified Chinese, Traditional Chinese, Japanese, English | None in current static snapshots | Chinese, Japanese, regional English source | None from current curated static sources | Short generic CJK group words remain disabled unless reviewed/corroborated |
| Mainland Southeast Asia | Cambodia, Laos, Myanmar, Thailand, Vietnam | Burmese, Thai, Lao, Khmer, Vietnamese, English | None in current static snapshots | Burmese, Thai, Lao, Khmer, Vietnamese, regional English source | None from current curated static sources | Generated/dictionary translations remain T5 and query-disabled by default |
| Maritime Southeast Asia | Brunei, Indonesia, Malaysia, Philippines, Singapore, Timor-Leste | Malay, Indonesian, English, Tagalog, Portuguese | None in current static snapshots | Malay, Indonesian, Tagalog, Portuguese, regional English source | None from current curated static sources | Generated/dictionary translations remain T5 and query-disabled by default |
| Caribbean introduced range | Cuba, Dominican Republic, Haiti, Jamaica, Puerto Rico | Spanish, English, Haitian Creole, French, Dutch, Papiamento | None in current static snapshots | Spanish, English, Haitian Creole, French, Dutch, Papiamento source | None from current curated static sources | Introduced-range names need source-backed regional evidence |
| Seychelles / Indian Ocean spread watch | Seychelles | Seychellois Creole, English, French | None in current static snapshots | Seychellois Creole, English, French source | None from current curated static sources | Watch-region names need occurrence support and source-backed regional evidence |
| Australia/New Guinea taxonomic caution | Australia, Papua New Guinea | English | None enabled by default | Reviewed English source with explicit accepted-taxon resolution | None by default | Regional names are blocked unless accepted-taxon resolution is explicit |

## Review Notes

- The registry can keep generated Hindi, Kannada, and other translation candidates for audit, but those rows remain T5 and query-disabled unless reviewed or independently corroborated.
- Source-backed Hindi or Kannada rows can become query-eligible when the static source row maps cleanly to the accepted GBIF species and is not ambiguous, broad-rank, or cautionary.
- Australia/New Guinea is deliberately cautionary because the configured workflow must not collapse unresolved Papilio demoleus / related-taxon evidence into global Papilio demoleus queries.
- A language target with no curated source row is a gap report item, not a request to invent a name.
