from annoy import AnnoyIndex
from tqdm import tqdm
import gensim.downloader
from pathlib import Path
import pickle

ANNOY_PATH = Path("./annoy/").resolve()


def download_model(model):
    ANNOY_PATH.mkdir(exist_ok=True)
    vectors = gensim.downloader.load(model)

    reduction_rate = 1 / 20
    reduced_vocab_size = int(vectors.vectors.shape[0] * reduction_rate)

    with open(ANNOY_PATH / f'{model}.pkl', 'wb') as f:
        pickle.dump(vectors.index_to_key[:reduced_vocab_size], f)

    t = AnnoyIndex(vectors.vectors.shape[1], 'angular')
    for i, vector in tqdm(enumerate(vectors.vectors[:reduced_vocab_size, :])):
        t.add_item(i, vector)
    t.build(10)
    t.save(str(ANNOY_PATH / f"{model}.ann"))


if __name__ == "__main__":

    models = list(gensim.downloader.info()['models'].keys())
    print(models)

    model_tqdm = tqdm(models[:1])
    for model in model_tqdm:
        model_tqdm.set_description(f"Processing model {model}")
        download_model(model)
