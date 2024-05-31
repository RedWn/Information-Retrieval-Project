import numpy as np
from tqdm import tqdm
from gensim import corpora, models
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Clusterer:
    def __init__(self, tfidf_matrix, nClusters):
        self.nClusters = nClusters
        self.svd = TruncatedSVD(n_components=nClusters)
        self.lsa_matrix = self.svd.fit_transform(tfidf_matrix)
        self.model = KMeans(n_clusters=nClusters).fit(self.lsa_matrix)
        self.topics_dict = {}

    def plot(self, size, topics) -> dict:
        pca = PCA(3)
        pca.fit(self.lsa_matrix)
        pca_matrix = pca.transform(self.lsa_matrix)

        fig, ax = plt.subplots(figsize=size)
        plt.style.use("classic")
        scatter = ax.scatter(
            pca_matrix[:, 0],
            pca_matrix[:, 1],
            c=self.model.labels_.astype(float),
            s=20,
            edgecolor="none",
        )
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="lower left", title="Topics"
        )
        ax.add_artist(legend1)
        for t, l in zip(
            legend1.texts,
            list(topics.values()),
        ):
            t.set_text(l)
        plt.show()

    def getTopics(self, dataset, keys) -> dict:
        topics_dict = {}
        for i in tqdm(range(0, len(self.svd.components_))):
            indices = np.where(self.model.labels_ == i)[0]
            text = [dataset[keys[x]] for x in indices]
            for doc in text:
                vocab = [[word] for word in doc]
            dictionary = corpora.Dictionary(word for word in vocab)
            corpus = [dictionary.doc2bow(word) for word in vocab]
            ldamodel = models.LdaModel(corpus, num_topics=1, id2word=dictionary)
            topics_dict[i] = ldamodel.print_topics(num_topics=1, num_words=5)[0][1]
        return topics_dict
