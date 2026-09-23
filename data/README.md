# Frozen data inputs

`manifest.json` specifies exact file sizes and SHA-256 values. The download
script uses the authors' already-public, commit-pinned mirror from the earlier
repository, including the Git LFS media endpoint for GIST. All four downloaded
files were hash-checked while preparing this release. Feature matrices are
excluded from the new repository.

Original dataset information:

* ISOLET: https://archive.ics.uci.edu/dataset/54/isolet
* CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
* Amazon Commerce Reviews: https://archive.ics.uci.edu/dataset/215/amazon+commerce+reviews+set

GIST here means the existing **Cifar10-Gist512.mat**, a 60,000 by 512 matrix.
It is not the different million-vector GIST benchmark. Its earliest extraction
recipe/license has not been independently recovered for this release. The
hash-locked author mirror preserves the exact experimental input; no new data
license or rights statement is asserted. Do not silently substitute another
GIST representation.

The first three original experiments use 2,800 training, 1,800 reference,
80 validation and 200 exposed test rows. Amazon uses 500/720/80/200.
The additional audits are 200 previously unused-at-freeze rows of each of the
first three matrices. They are not external datasets. Amazon has no audit rows.
Preprocessing is fitted only to training rows. The frozen bandwidths and radii
are in `configs/datasets/*/metadata.json`; row IDs are in `split_ids.json`.
