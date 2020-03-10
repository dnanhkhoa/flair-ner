# -*- coding: utf-8 -*-
from flair.datasets import CONLL_03, DataLoader
from flair.models import SequenceTagger

if __name__ == "__main__":
    corpus = CONLL_03(base_path="data/conll-2003")

    tagger = SequenceTagger.load("models/en-ner-conll03-v0.4.pt")

    dev_eval_result, dev_loss = tagger.evaluate(
        DataLoader(corpus.test, batch_size=64, num_workers=8)
    )

    print(dev_eval_result.main_score)
