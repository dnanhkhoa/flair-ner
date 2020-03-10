# -*- coding: utf-8 -*-
from flair.datasets import CONLL_03
from flair.embeddings import PooledFlairEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus = CONLL_03(base_path="data/conll-2003")

embedding_types = [
    WordEmbeddings("glove"),
    PooledFlairEmbeddings("news-forward", pooling="min"),
    PooledFlairEmbeddings("news-backward", pooling="min"),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=corpus.make_tag_dictionary(tag_type="ner"),
    tag_type="ner",
)


trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train("models/checkpoints", train_with_dev=True, max_epochs=150)
