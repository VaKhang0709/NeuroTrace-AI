# Genuine recorded UCI395 excerpts — not generated numeric fixtures

Attribution: Isenkul, M., Sakar, B., & Kursun, O. (2014). *Parkinson Disease Spiral
Drawings Using Digitized Graphics Tablet* [Dataset]. UCI Machine Learning
Repository. DOI: **10.24432/C5Q01S**, CC BY 4.0 (dataset attribution; this repository's
software license does not replace it). Canonical catalog:
https://archive.ics.uci.edu/dataset/395/ . License/citation cross-reference:
https://github.com/oleeveeuh/parkinsons-drawing-analysis/blob/main/docs/data_sources.md .

These small leading excerpts of already public pseudonymized research recordings
contain no participant names,contact information or full study trajectories. No
new private Firebase/mobile data are included. Treat the clinical metadata
sensitively; no clinical use is supported. Files stay raw-format test evidence,
never an invented dataset or additional subjects.

The pinned 2020 GitHub mirror and original-file hashes/line ranges are listed in
PROVENANCE.json. Each original blob was retrieved through GitHub's API and its Git
blob SHA-1 checked before taking an excerpt. Numeric rows are UNCHANGED. Four
DIFFERENT recorded source files are used; no path was cloned to invent participants.
LF normalization and truncation are documented. C_0002 adds the literal column
header from readme.txt to exercise named-header parsing; the original
trajectory is HEADERLESS. It does NOT pretend that the original had that header.

Verified columns: **X;Y;Z;Pressure;GripAngle;Timestamp;Test ID**. The README does
NOT specify clock units,physical coordinate calibration or an azimuth channel.
Tests pass explicit `milliseconds` as a documented interpretation,not a discovery.
The six-column Azimuth/Tilt header in the Kaggle report was NOT observed in these
originals. It is tested separately as a named-header adapter using reprojected
recorded columns,not called an untouched source file. GripAngle is never azimuth.

Full real recordings used for local integration remain outside Git under /tmp;
only these bounded public excerpts are committed. Never use this fixture subset
for reporting disease performance or for claiming independent clinical validation.
