
from pq.main import PairCorpus, QualityEstimator, TextDegradation
from pq.similarity_measures import TfidfCosineSimilarity


corpus = PairCorpus(
    source_path="files/corpus/eng-engULB.txt",
    target_path="files/corpus/kos-kos.txt",
)

estimator = QualityEstimator(
    similarity_measures=[TfidfCosineSimilarity()],
    combination_mode="multiply",
)

pair_index = 100
random_pair = corpus.get_pairs(pair_index)


print("Source: ", random_pair[0])
print("Target: ", random_pair[1])
print("Quality: ", estimator.evaluate_translation(random_pair[0], random_pair[1], corpus, sample_size=25))






