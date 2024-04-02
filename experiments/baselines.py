from pathlib import Path
import random

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring

from util import legal_sent_tokenize


def eval_baseline(
    articles, abstracts, token_budget=None, sent_budget=None, randomize=False
):
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    aggregator = scoring.BootstrapAggregator()

    sents_per_summary = []
    tokens_per_summary = []
    tokens_per_abstract = []
    scores = {}
    summaries = []

    if token_budget is None and sent_budget is None:
        token_budget = [
            sum([len(sent.split()) for sent in abstract]) for abstract in abstracts
        ]
        token_budget = int(np.mean(token_budget))
        print(f"Using a budget of {token_budget} tokens")

    for article, abstract in tqdm(zip(articles, abstracts)):
        sents = []
        tokens = 0

        if not isinstance(article, list):
            article = legal_sent_tokenize(article)

        if not isinstance(abstract, list):
            abstract = [abstract]

        if randomize:
            article = random.sample(article, len(article))

        for sent in article:
            sent_tokens = len(sent.split())

            if token_budget and tokens >= token_budget:
                break

            if sent_budget and len(sents) >= sent_budget:
                break

            sents.append(sent)
            tokens += sent_tokens

        sents_per_summary.append(len(sents))
        tokens_per_summary.append(tokens)
        abstract_tokens = sum([len(s.split()) for s in abstract])
        tokens_per_abstract.append(abstract_tokens)

        score = rouge.score("\n".join(abstract), "\n".join(sents))
        aggregator.add_scores(score)
        summaries.append("\n".join(sents))

    print("Avg sentences per summary:", np.mean(sents_per_summary))
    print("Avg tokens per abstract:", np.mean(tokens_per_abstract))
    print("Avg tokens per summary:", np.mean(tokens_per_summary))
    print()

    scores = aggregator.aggregate()

    for k, v in sorted(scores.items()):
        print("%s-R,%f,%f,%f" % (k, v.low.recall, v.mid.recall, v.high.recall))
        print("%s-P,%f,%f,%f" % (k, v.low.precision, v.mid.precision, v.high.precision))
        print("%s-F,%f,%f,%f\n" % (k, v.low.fmeasure, v.mid.fmeasure, v.high.fmeasure))

    return scores, summaries


def run(
    data_path,
    token_budget=100,
    source="text",
    target="summary",
    random=True,
    output_path=None,
):
    data = pd.read_csv(data_path)
    paragraphs_path = Path(data_path).parent / f"paragraphs_{Path(data_path).name}"
    
    if Path(paragraphs_path).exists():
        paragraphs = pd.read_csv(paragraphs_path)
        print(11, paragraphs.columns)
        data["oracle_paragraphs"] = paragraphs["oracle_paragraphs"]

    data = data.fillna("")
    _, summaries = eval_baseline(
        data[source], data[target], token_budget=token_budget, randomize=random
    )
    if output_path:
        output_data = {"prediction": summaries, "reference": data[target]}
        pd.DataFrame(output_data).to_csv(output_path, index=None)


if __name__ == "__main__":
    fire.Fire(run)
