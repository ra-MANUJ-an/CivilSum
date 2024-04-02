import logging
from pathlib import Path
import pickle
import sys
sys.path.append('factorsum')

import evaluation.factorsum as factorsum_evaluation
from factorsum.guidance import ROUGEContentGuidance
from factorsum.oracle import get_oracles
from factorsum.sampling import sample_dataset
import fire
import pandas as pd
import nltk
from rich.logging import RichHandler
from tqdm import tqdm

from util import fix_sent_tokenization, legal_sent_tokenize

logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.to_dict(orient='records')
    data = [x for x in data if str(x['text']) != 'nan']
    return data


def preprocess(
    dataset,
    source_key="article",
    target_key="abstract",
    min_words=10,
    oracle_type="rouge",
    include_oracles=False,
):
    """
    Preprocess a dataset with source/target texts.
    """
    sources = []
    targets = []
    
    def get_sentences(text):
        paragraphs = text.split("\n")
        sents = []

        for par in paragraphs:
            par_sents = nltk.sent_tokenize(par)
            par_sents = fix_sent_tokenization(par_sents, min_words)
            par_sents = [s.strip() for s in par_sents]
            sents.extend(par_sents)
        return sents

    for idx, item in tqdm(enumerate(dataset)):
        if isinstance(item[source_key], float) or len(item[source_key]) == 0:
            logger.warning(f"{idx}: empty article found!")
            logger.warning("Target text:")
            logger.warning(item[target_key])
            continue

        sources.append(get_sentences(item[source_key]))
        targets.append(get_sentences(item[target_key]))

    logger.info(f"Preprocessed {len(sources)} samples")
    data = {
        "sources": sources,
        "targets": targets,
    }

    if include_oracles:
        logger.info("Collecting oracles...")
        oracles = get_oracles(
            sources, targets, min_words=min_words, oracle_type=oracle_type
        )
        data["oracles"] = oracles

    return data


def save_dataset_samples(data_path, split, sample_fn=None, sample_factor=5, 
                         samples_per_doc=20, require_oracle=True):
    data = load_data(data_path)
    data = preprocess(data, source_key='text', target_key='summary', include_oracles=True)
    processed_data_path = str(data_path).replace("csv", "pkl")
    with open(processed_data_path, 'wb') as fh:
        pickle.dump(data, fh)

    views, _ = sample_dataset(data, sample_factor=sample_factor,
                                views_per_doc=samples_per_doc,
                                sample_fn=sample_fn, verbose=False,
                                require_oracle=require_oracle)

    print(f'sample factor: {sample_factor}; num samples: {len(views["sources"])}')

    views_path = Path(data_path).parent / f'random_k_{sample_factor}_samples_{samples_per_doc}_{split}.pkl'
    with open(views_path, 'wb') as fh:
        pickle.dump(views, fh)

    

def preprocess(data_folder):
    for split in ["train", "validation", "test"]:
        data_path = Path(data_folder) / f"{split}.csv"
        if data_path.exists():
            require_oracle = split == "train"
            save_dataset_samples(data_path, split, require_oracle=require_oracle)


def evaluate(
    doc_id=None,
    max_samples=10000,
    data_dir="data",
    dataset_name="legal",
    split="test",
    training_domain="legal",
    source_token_budget=None,
    token_budget=160,
    intrinsic_model_id=None,
    content_type=None,
    budget_type='fixed',
    method="factorsum",
    views_per_doc=20,
    sample_factor=5,
    min_words_per_view=5,
    model_path=None,
    seed=17,
    **kwargs,
):
    sent_tokenize_fn = None
    model_kwargs = None
    if token_budget is None:
        budget_type = None

    model_kwargs = {}
    model_kwargs["intrinsic_importance_model_path"] = model_path
    
    if training_domain == 'legal':
        model_kwargs['intrinsic_importance_model_id'] = 'e10y5ct4'
        model_kwargs['intrinsic_importance_model_url'] = 'https://www.dropbox.com/scl/fi/tr31ocahdv7m0qyw9tzww/model-e10y5ct4.zip?rlkey=5p31bos2u8odq5kurlsdzboj4&dl=1'
    
    if dataset_name == 'legal':
        sent_tokenize_fn = legal_sent_tokenize

    custom_guidance = None
    if dataset_name == 'legal' and content_type == 'oracle':
        data_path_ = Path(data_dir) / f"paragraphs_{split}.csv"
        par_data = pd.read_csv(data_path_)
        par_guidance = []
        for idx, row in par_data.iterrows():
            text = row['oracle_paragraphs']
            if str(text) == 'nan':
                text = ''
            par_guidance.append(ROUGEContentGuidance(text))

    elif content_type == 'oracle':
        raise ValueError(f"Content guidance type not supported: {content_type}")

    factorsum_evaluation.evaluate(
        doc_id=doc_id,
        max_samples=max_samples,
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=split,
        training_domain=training_domain,
        source_token_budget=source_token_budget,
        token_budget=token_budget,
        intrinsic_model_id=intrinsic_model_id,
        content_type=content_type,
        budget_type=budget_type,
        custom_guidance=custom_guidance,
        method=method,
        views_per_doc=views_per_doc,
        sample_factor=sample_factor,
        min_words_per_view=min_words_per_view,
        sent_tokenize_fn=sent_tokenize_fn,
        model_kwargs=model_kwargs,
        seed=seed,
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire()
    