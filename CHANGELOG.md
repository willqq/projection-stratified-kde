# Release preparation

The final scientific estimator and saved observations are unchanged.

* Collected the base experiment source/assets and later supplemental drivers.
* Exposed a projection-only Python API using two byte-identical frozen modules.
  The wrapper removes legacy MECH initialization and is checked against the
  frozen evaluator; it is not the historical timing implementation.
* Added hash-checked dataset acquisition and frozen-split statistical replay.
* Added table reconstruction from original query records, preserving query-level
  bootstrap and the final stronger DEANN selection.
* Kept complete supplemental records in separate release attachments.
* Added source provenance, verification scope and third-party licensing notes.

No experiment configurations were reselected, and no paper result was replaced.
No GitHub repository, release or online submission has been created by this step.
