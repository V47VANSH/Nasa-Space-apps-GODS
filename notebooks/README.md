# Original hackathon notebooks

These are the NASA Space Apps 2024 submission, kept as a historical record of
how the approach was developed.

| Notebook | Contents |
|---|---|
| `01_training_and_feature_engineering.ipynb` | Feature extraction from the Apollo 12 catalog and model training |
| `02_inference_walkthrough.ipynb` | STA/LTA detection and prediction on a test record |

**They are not maintained and no longer run as written.** `np.asfarray` was
removed in NumPy 2.0, and the paths point at a data layout that predates the
current repository structure. The working implementation is `src/cosmoquake/`.

They also contain four defects that the package fixes — a wrong label mapping,
sample indices compared against seconds, hardcoded scaler constants, and the
NumPy call above. See the "What changed in v1.0" section of the top-level
README for details.

To reproduce the original work, use the package instead:

```bash
cosmoquake train    # equivalent to notebook 01
cosmoquake predict Resources/Data/mars/test/data/*.csv   # equivalent to 02
```
