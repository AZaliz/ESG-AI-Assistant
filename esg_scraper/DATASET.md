# PDF Dataset Snapshot

The raw ESG / CSRD-ESRS PDF corpus consumed by this scraper is **not stored in
git** — it is ~3.3 GB uncompressed and ships as a ~2.24 GB zip, which exceeds
GitHub's 100 MB per-file limit (and the 2 GB Git LFS / Release-asset caps).
It is distributed out-of-band instead.

## Snapshot: `data_pdfs_snapshot_2026-05-19.zip`

| Field      | Value |
|------------|-------|
| Size       | 2,239,055,749 bytes (~2.24 GB) |
| SHA-256    | `ab5513eb3e892250fe922ab5828a7c68d65c729b6cc8ddb3dff67c991f31a0c3` |
| Created    | 2026-05-19 |
| Expands to | `esg_scraper/data/` (raw company PDFs + scraped sources) |

## Download

> **Link:** _<!-- TODO: paste the external share URL (Google Drive / etc.) here -->_

## Verify & extract

```bash
# from the esg_scraper/ directory
sha256sum -c data_pdfs_snapshot_2026-05-19.zip.sha256   # must print: OK
unzip data_pdfs_snapshot_2026-05-19.zip -d .            # -> esg_scraper/data/
```

On Windows / PowerShell:

```powershell
(Get-FileHash data_pdfs_snapshot_2026-05-19.zip -Algorithm SHA256).Hash.ToLower()
# compare against the SHA-256 above, then:
Expand-Archive data_pdfs_snapshot_2026-05-19.zip -DestinationPath .
```

`data/` is git-ignored, so extracting in place will not pollute the working
tree. Anyone reproducing the pipeline only needs this snapshot to populate the
inputs the scraper and parser expect.
