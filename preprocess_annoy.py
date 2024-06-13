from annoy import AnnoyIndex
from tqdm import tqdm
import gensim.downloader
from pathlib import Path
import pickle

ANNOY_PATH = Path("./annoy/").resolve()


if __name__ == "__main__":
    ANNOY_PATH.mkdir(exist_ok=True)

    models = list(gensim.downloader.info()['models'].keys())
    print(models)

    model_tqdm = tqdm(models[2:3])
    for model in model_tqdm:
        model_tqdm.set_description(f"Processing model {model}")
        vectors = gensim.downloader.load(model)

        with open(ANNOY_PATH / f'{model}.pkl', 'wb') as f:
            pickle.dump(vectors.index_to_key, f)

        t = AnnoyIndex(vectors.vectors.shape[1], 'angular')
        for i, vector in tqdm(enumerate(vectors.vectors)):
            t.add_item(i, vector)
        t.build(10)
        t.save(str(ANNOY_PATH / f"{model}.ann"))