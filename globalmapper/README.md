# GlobalMapper replication

Preliminary reproduction of GlobalMapper on its released
10K-block dataset.

## Training
- Model: GlobalMapper, 50 epochs completed.
- Original dataset split: 8,000 training / 2,000 validation.
- Checkpoint epoch: 50.
- Training metrics: `training/training_history.csv`.
- Training curves: `training/training_progress.png`.
- Configuration: `training/config.yaml`.
- Split membership: `training/split_indices.npz`.
- Block 1824 belongs to the validation split.

## Weights
`checkpoints/globalmapper_epoch50_weights.pt` is a model-only
state_dict, to be uploaded using Git LFS. The complete
optimizer-containing checkpoint remains in Google Drive.

## Limitations
- Training was performed on GlobalMapper's released dataset,
  not the team NYC training split.
- Empirical posterior-mean latent sampling differs from the
  official sampling protocol.
- NYC adaptation, 4 m raster conversion and shared evaluation
  remain unfinished.
- Preliminary single-block geometry results are not comparable
  to the team's NYC benchmark.
