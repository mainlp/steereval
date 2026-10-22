import os
import numpy as np

from constants import LANG2LANGKEY


def get_vector_path(lang, max_pairs, data, model_name):
    model_name = model_name.replace("/", "-")
    return f"./cache/{data}/intervention_vectors/{LANG2LANGKEY[lang]}_to_en_{model_name}_{max_pairs}_residual.npy"


def load_vectors(lang, max_pairs, data, model_name):
    path = get_vector_path(lang, max_pairs, data, model_name)
    if os.path.exists(path):
        print("Loading existing intervention vectors for", lang)
        Vs = np.load(path, allow_pickle=True)
        if len(Vs) > 0:
            return Vs
    return None


def save_vectors(Vs, lang, max_pairs, data, model_name):
    cache_path = get_vector_path(lang, max_pairs, data, model_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print("Saving intervention vectors for", lang)
    np.save(cache_path, Vs)
