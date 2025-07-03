# Causal graph comparison 

## Installation

**Requires python version 3.11 and the `poetry` management tool**

To install poetry : https://python-poetry.org/docs/#installation

```python
poetry install --with dev
```

To check QA

```python
nox -s precommit
```

## Usage instructions

## TODO
- [x] Make MLP into reusable module
- [ ] Write sh script to run 03-train-mlp.py and 04-test-mlp.py on cluster
- [ ] Edit run 03-train-mlp.py and 04-test-mlp.py to run on cluster
- [ ] Replace all vars with config

- [ ] run PCMCI+ from tigris package on mlp output
- [ ] run RMSE on mlp output vs ground truth output
- [ ] run causal graph comparison on mlp output vs ground truth output SAVAR

