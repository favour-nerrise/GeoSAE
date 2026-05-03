# Sample Data

This directory contains a tiny synthetic example that shows the expected public
repository structure.

It is included only for demonstration:

- the embeddings are random,
- the labels are illustrative,
- the dataset is far too small for meaningful training or evaluation.

Files:

- `metadata.csv`: scan-level metadata.
- `mci_conversion_labels.csv`: one conversion label per subject.
- `features/layer_09/*.npy`: one embedding vector per scan.

The scan stems in `metadata.csv` match the `.npy` filenames under
`features/layer_09/`.

