# V6 limitations and nonclaims

- **No real V6 training or clinical validation occurred.** No held-out AUC,
  sensitivity, specificity, ECE, mobile-PD sensitivity or SOTA superiority exists.
  >0.97 AUC is a research target, not a result or release criterion to game.
- No Firebase credentials/authorized submissions or real external datasets are
  mounted; license/access instructions could not be verified behind the network
  restriction. Gated adapters are placeholders for approved conversion, not native
  loaders magically able to parse unknown data dictionaries.
- The historical V5 cohort is small and potentially device/label-confounded.
  Mobile-HC-only data cannot establish mobile-PD sensitivity. Formula paths and
  injected tremor do not resolve this identifiability problem.
- Synthetic tremor is not Parkinson disease. Injection can create a frequency
  shortcut rather than improve disease discrimination. It defaults OFF, requires
  train-only donor/recipient parents and must be separately ablated. No synthesized
  “PD” example is allowed in clinical evaluation or calibration.
- FiLM explicitly exposes device identity, conflicting with the aim of device
  invariance. The frozen-feature probe may fail; failure must be reported, not
  tuned away on test. Age auxiliary targets can amplify confounding and require
  age-balanced analysis, missingness review and no-age ablations.
- Per-window z-score normalization suppresses some absolute amplitude/pressure
  information; db4 band reconstruction and cubic interpolation can introduce
  boundary artifacts. Canonical rates/filters are priors requiring real acquisition
  validation. The V6 contract is not interchangeable with V5.
- Raw strokes, coordinates, timestamp units, azimuth radians, diagnosis provenance
  and stable human identities require upstream verification. Device UUID alone
  may split one participant across devices or merge multiple users on a shared one.
  Cross-dataset alias linkage and consent cannot be asserted from a filename.
- Fivefold GroupKFold is not stratified. Folds with one label abort AUC selection.
  Fivefold models must not ensemble-predict their own development participants.
  Frozen encoder device probing must use parents absent from all backbone training.
- Bootstrap estimates from a small cohort can be unstable; undefined one-class
  AUC replicates are excluded and reported. McNemar tests classifications at0.50,
  not AUC superiority; multiple exploratory comparisons require correction and a
  prespecified independent external test before publication.
- Training-loss SSL plateau is not a held-out clinical validation signal. Shared
  self-supervision over validation/test subjects is prohibited in this protocol.
- Calibration on the same small member-validation set used for model selection
  is prone to optimism. Ensemble calibration requires a separate all-member-unseen
  cohort. Neither a real isotonic map nor a calibrated patient risk is delivered.
- CPU correctness succeeded; A100/T4 memory, speed, long-run stability,150–300epoch
  convergence and five-model browser feasibility are NOT measured. Approximate
  VRAM guidance is a starting point, not a benchmark.
- Current ingestion is append-protected at each task output; reruns verify existing
  manifest checksums and skip immutable task keys. Remote edits need new task IDs. Cloud connectivity,
  permissions, unexpected schemas and SDK errors have not been integration-tested.
- No real pretrained/fine-tuned V6 checkpoints or download links exist. Export
  smoke weights are random initialization or tiny synthetic-test models and are
  deliberately not published as deployable artifacts. V5 files remain preserved.
