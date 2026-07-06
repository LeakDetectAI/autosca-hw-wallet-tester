import h5py
import numpy as np


class ASCADAttackScores:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_attack_scores(self, model_name, key, plaintext, ciphertext, mask, offset):
        results = {}
        with h5py.File(self.file_path, 'r') as hdf:
            for key in hdf.keys():
                results[key] = np.array(hdf[key])
        return results
