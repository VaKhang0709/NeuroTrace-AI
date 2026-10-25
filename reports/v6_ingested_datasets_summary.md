# V6 public dataset acquisition summary

Machine-recorded acquisition outcomes; no real sample counts inferred from failed requests.

## UCI395

```json
{
  "dataset": "UCI395",
  "status": "BLOCKED",
  "detail": "URLError: <urlopen error TLS/SSL connection has been closed (EOF) (_ssl.c:992)>",
  "fallback": "--local-uci /authorized/uci.zip OR /authorized/raw_dir"
}
```

## Zenodo/Figshare/PhysioNet

```json
{
  "dataset": "Zenodo/Figshare/PhysioNet",
  "status": "NO_VERIFIED_DOI_SUPPLIED",
  "detail": "No automatic search result is assumed to be licensed dynamic handwriting. Supply reviewed mirror manifest."
}
```

Only acquisition is reported here. Raw units, participant IDs, labels and licensing still require source review.

## Separate engineering scale test — NOT an ingested real dataset

10002mock-source synthesized trajectories were materialized under`/tmp`
(5001touch-target+5001mouse-target;3formula parents;0real humans).
They are taggedMOCK_SOURCE and rejected by production training. Real recorded-source
synthesis remains blocked. This count must not be added to real participant totals.
