# :lemon::dizzy: **LEMONsc: Learning the Evolution of Massive Objects in Nuclear star clusters**

---
## :computer::open_file_folder: **Scripts available:**
```
scripts/
    │_ get_features.py      # Explore and prepare relevant tabular features.
    │_ mltrees_training.py  # Optimize, train and predict classical ML models.
```
---

## :question: **How to run:**

Asumming that you are inside the container or that you possees the require libraries. We present the following scripts:

### **$\bullet$ get_features.py**

This script process and prepares features from the selected dataset (so far, created and tested for *moccasurvey* dataset). This script has three use-modes:

```
python3 -m scripts.get_features --mode [study|feats|plot]
```

- `study`: Analize the available simulations and generate vizualization of the dataset characteristics.
- `feats`: Generate a dataframe with the features and partitions for ML-Training.
- `plot`: Create vizualization of the generated features.

### **$\bullet$ mltrees_training.py**

This script optimize and trains MLTreeRegressor models used a selected dataset. . This script has three use-modes:

```
python3 -m scripts.mltrees_training --mode [optim|train|predict]
```

- `optim`: Do a space search of hiperparameters using cross-validation with Optuna.
- `train`: Train a regressor and evaluate the performance and unceirtanty.
- `predict`: (*work in progress*) Load training models and do inference.

---
