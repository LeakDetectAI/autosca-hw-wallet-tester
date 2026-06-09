import copy

import keras_tuner as kt
from keras_tuner.engine import hyperparameters as hp

from deepscapy.experimental_utils import loss_dictionary_attack_models


class HPOModelLoss(kt.HyperModel):
    """HyperModel that searches over loss-function hyperparameters only."""

    def __init__(self, learner, learner_params, hp_dict, lf_name):
        super().__init__()
        self.learner = learner
        self.learner_params = copy.deepcopy(learner_params)
        self.hp_dict = hp_dict
        self.lf_name = lf_name

    def build(self, hp):
        params = {}
        for key, values in self.hp_dict.items():
            params[key] = hp.Choice(key, values)
        loss_fn = loss_dictionary_attack_models[self.lf_name](**params)
        learner_params = copy.deepcopy(self.learner_params)
        learner_params["loss_function"] = loss_fn
        return self.learner(**learner_params).model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, validation_data=validation_data, **kwargs)
        return model


class HPOModelTuner(kt.Tuner):
    """Tuner that uses random search over loss hyperparameters."""

    def run_trial(self, trial, x, y, validation_data, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        model.fit(x, y, validation_data=validation_data, **kwargs)
        val_loss, val_acc = model.evaluate(validation_data[0], validation_data[1], verbose=0)
        self.oracle.update_trial(trial.trial_id, {"val_accuracy": val_acc})
        self.save_model(trial.trial_id, model)
