import os
import numpy as np
from constants import LANG2LANGKEY


def get_map_path(lang, max_pairs, data, model_name):
    model_name = model_name.replace("/", "-")
    return f"cache/{data}/layer_maps/{LANG2LANGKEY[lang]}_to_en_{model_name}_{max_pairs}_residual.npy"


def load_layer_map(lang, max_pairs, data, model_name):
    path = get_map_path(lang, max_pairs, data, model_name)
    if os.path.exists(path):
        print("Loading existing numpy layer maps for", lang)
        Ws = np.load(path, allow_pickle=True)
        if len(Ws) > 0:
            return Ws

    old_path = path.replace(data, f"gptscore/{data}")
    if os.path.exists(old_path):
        # backwards compatible for when we saved in gptscore folder, don't do
        # the same for geval because of slight difference in sampling
        print("Loading existing numpy layer maps for", lang)
        Ws = np.load(old_path, allow_pickle=True)
        if len(Ws) > 0:
            return Ws
    return None


def save_layer_map(Ws, lang, max_pairs, data, model_name):
    path = get_map_path(lang, max_pairs, data, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Saving layer maps for", lang)
    np.save(path, Ws)
