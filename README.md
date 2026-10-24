# scratch_ml

Classical machine learning algorithms implemented **from scratch** using only
**NumPy** and the Python standard library. No scikit-learn, SciPy, or other ML
frameworks.

The estimators follow the scikit-learn style: `fit(X, y)`, `predict(X)`,
`predict_proba(X)` where it makes sense, and fitted attributes ending in `_`.

## Algorithms

| Module | Classes |
| --- | --- |
| `linear_model` | `ScratchLinearRegression`, `ScratchLogisticRegression` |
| `neighbors` | `ScratchKNN` |
| `naive_bayes` | `ScratchNaiveBayes` (Gaussian) |
| `svm` | `ScratchLinearSVM` (hinge loss + gradient descent) |
| `tree` | `ScratchDecisionTree` (gini / entropy / mse) |
| `ensemble` | `ScratchRandomForest`, `ScratchGradientBoosting` |
| `cluster` | `ScratchKMeans` |
| `decomposition` | `ScratchPCA` (eigendecomposition, with an SVD cross-check) |
| `metrics` | `accuracy_score`, `mean_squared_error`, `mean_absolute_error`, `r2_score` |
| `datasets` | `make_regression`, `make_classification`, `make_blobs`, `make_moons` |
| `utils` | `train_test_split` |

## Installation

```bash
git clone https://github.com/<your-username>/scratch_ml.git
cd scratch_ml
pip install -r requirements.txt
```

The only dependency is NumPy. There is no `setup.py` yet, so use the package
from the project root.

## Quick start

```python
from scratch_ml import (
    ScratchRandomForest,
    make_moons,
    train_test_split,
    accuracy_score,
)

X, y = make_moons(n_samples=500, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = ScratchRandomForest(n_estimators=25, max_depth=8, random_state=0)
model.fit(X_train, y_train)

print(accuracy_score(y_test, model.predict(X_test)))  # ~0.99
```

## Running the demos

Every module has a synthetic-data demo. The package uses relative imports, so
run these with `-m` from the project root:

```bash
python -m scratch_ml          # run every model's demo
python -m scratch_ml.tree     # run a single module's demo
```

## Project layout

```
scratch_ml/
├── __init__.py
├── __main__.py        # runs all demos
├── base.py            # BaseEstimator, NotFittedError
├── linear_model.py
├── neighbors.py
├── naive_bayes.py
├── svm.py
├── tree.py
├── ensemble.py
├── cluster.py
├── decomposition.py
├── metrics.py
├── datasets.py
└── utils.py
```

## Notes

This is a learning project. The code favours readability over speed and is
not meant to replace scikit-learn for real workloads.

## License

MIT. See [LICENSE](LICENSE).
