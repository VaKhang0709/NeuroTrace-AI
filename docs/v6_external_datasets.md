# V6 external dataset access audit — 2026-10-24

**No new real dataset downloaded or integrated in this environment.** Direct HTTPS
attempts to the requested sources failed at the sandbox network boundary (curl
SSL_ERROR_SYSCALL for UCI/UNESP/PhysioNet/Synapse, DNS resolution failure for BDALab). Outbound access is restricted to GitHub and package registries.
A failed network request is NOT proof a dataset is unavailable at its institution.
No raw UCI source directory is mounted locally, despite previous-session availability.
V5 weights are not a substitute for UCI trajectories. Existing reported counts are
not independently reverified here. No fabricated data or invented access approvals.

| Source / attempted URL | Current status | Loader |
|---|---|---|
| UCI #395 https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet | TLS/network blocked; local raw unavailable | `uci_loader.py`, audited V5 adapter |
| PaHaW https://bdalab.utko.feec.vutbr.cz/ | Network blocked; license/access not verified | `pahaw_loader.py`, gated unified-schema adapter |
| HandPD https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/ | Network blocked; native modality/access not verified | `handpd_loader.py`, gated adapter |
| NewHandPD | Exact current official download URL UNKNOWN; follow HandPD institution portal, do not guess a mirror | `newhandpd_loader.py`, gated adapter |
| PhysioNet https://physionet.org/content/?topic=parkinson | Search request blocked; no Parkinson handwriting dataset confirmed | `physionet_loader.py`, gated adapter |
| mPower https://www.synapse.org/Synapse:syn4993293 | Synapse request blocked; access/record schema unverified | `mpower_loader.py`, gated adapter |

## Operator application procedure (institutional details must be verified)

The exact current application form, staff email and licensing text could NOT be
verified behind the network restriction. The following are concrete operator steps,
not a claim these institutions approved access or that a particular form is current:

1. Open the official URL outside this sandbox; record landing page/date/license.
2. For PaHaW, locate dataset/access instructions on the BDALab portal. If a
   research agreement/application is required, submit institutional affiliation,
   research purpose and PI details via the portal's current official contact/form.
   Retain signed permission locally and request raw digitizer `.svc` streams,
   participant mapping, acquisition units and variable dictionary.
3. For HandPD/NewHandPD, use the official UNESP dataset landing page/contact to
   request each dataset's access terms and raw dynamic trajectory availability.
   If only images are supplied, mark ineligible for trajectory training; do NOT
   reconstruct kinematics from pixels. Verify NewHandPD's current official location.
4. On PhysioNet, search “Parkinson handwriting”, “spiral”, and “digitizer”; check
   actual modalities, subject IDs, labels and access terms before adding an adapter.
   Current search findings here are UNKNOWN, not “none exist.”
5. For mPower, create an authorized Synapse account and follow the current study's
   research-use governance/certification/data-use agreement requirements. Verify
   the official study page and request permitted raw touchscreen events. Tapping
   is a different task, not a handwriting PD target. Restrict to separately tracked
   weak/self-supervision unless cross-task/participant provenance is established.
6. Store approved originals under ignored `data/raw/`, add checksums and private
   license records, validate patient IDs and units; convert to explicit unified
   JSON before using gated adapters. Missing diagnosis stays -1 (unlabeled).

Cross-source same-person linkage, labels, consent and held-out partitions must be
reviewed before pooling. Device UUID is not guaranteed unique human identity.
