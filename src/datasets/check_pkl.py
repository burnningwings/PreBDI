from generate_training_data import load_pkl
import pandas as pd
import numpy as np

if __name__ == "__main__":
    filepath = "src/data/PlanarBeamSplitCaseSingle/PlanarBeam_{}.pkl"

    train_data = load_pkl(filepath.format("train"))
    eval_data = load_pkl(filepath.format("eval"))
    test_data = load_pkl(filepath.format("test"))

    train_label = train_data["label"]
    eval_label = eval_data["label"]
    test_label = test_data["label"]

    train_damage_label = np.any(train_label > 30, axis=0)
    eval_damage_label = np.any(eval_label > 30, axis=0)
    test_damage_label = np.any(test_label > 30, axis=0)

compare = [train_damage_label, eval_damage_label, test_damage_label]

print("end-----------")
