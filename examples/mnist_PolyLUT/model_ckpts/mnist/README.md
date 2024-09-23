# Ensembling checkpoints

Each model has an `hparams.yml` file, which has the model configuration and the hyperparameters used to train it.
You'll need this file to load the models.

The ensemble checkpoints for the `./averaging/` models are stored as `best_accuracy.pth`, whereas all other ensemble checkpoints are stored as `last_ensemble_ckpt.pth`.
