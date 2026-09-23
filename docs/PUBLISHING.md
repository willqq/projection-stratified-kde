# Publishing this prepared repository

Suggested repository name: `projection-stratified-kde`.

Suggested description:
`Projection-guided remainder stratification for high-dimensional Gaussian kernel sums: code, frozen protocols and reproducibility artifacts.`

Create an empty GitHub repository under the author's chosen account. Do not
initialize another README, .gitignore or license in the GitHub web form; the
prepared tree already contains the documentation. Resolve the author-code
license before a public release. The repository can be created private while
the final release checks are completed.

Upload the prepared source tree using Git. Upload `experiment-records.tar.gz`
and `upgrade-raw-records.tar.gz` as release attachments rather than committing
them as large Git objects. Their SHA-256 values are in
`provenance/RELEASE_ASSETS.json`. After the actual upload, record the real download
URLs and tag/commit in the README; this package does not invent those URLs.

Retain the earlier `mech-annulus-kde` repository as historical work. The new
repository explains the final projection-based method and keeps the old
norm-angle implementation under the frozen experiment sources.

Before publication, confirm the two authors' permission to license their code.
Third-party DEANN keeps its MIT license. The GIST feature matrix is downloaded
from the existing hash-locked mirror; its original extraction provenance remains
a documented data limitation. No original feature matrices are uploaded as part
of this new code tree.

The full manuscript PDF and author metadata are not automatically published with
the code. Add a manuscript link only after the authors choose the public version.
