import numpy as np
from tqdm import tqdm
from gensim import corpora, models
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud


class Clusterer:
    def __init__(self, tfidf_matrix, nClusters):
        self.nClusters = nClusters
        self.svd = TruncatedSVD(n_components=nClusters)
        self.lsa_matrix = self.svd.fit_transform(tfidf_matrix)
        self.model = KMeans(n_clusters=nClusters).fit(self.lsa_matrix)
        self.topics_dict = {}

    def plot(self, size) -> dict:
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
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="lower left", title="Classes"
        )
        ax.add_artist(legend1)
        plt.show()

    def getTopics(self, dataset, keys) -> dict:
        topics_dict = {}
        if self.nClusters > 6:
            fig, axs = plt.subplots(3, 4)
        else:
            fig, axs = plt.subplots(2, 3)
        fig.set_figwidth(21 * 3)
        fig.set_figheight(21)
        for i in tqdm(range(0, self.nClusters)):
            indices = np.where(self.model.labels_ == i)[0]
            text = [dataset[keys[x]] for x in indices]
            for doc in text:
                vocab = [word for word in doc]
            x = " ".join(vocab)
            wordcloud = WordCloud(
                width=800,
                height=800,
                background_color="white",
                stopwords=None,
                min_font_size=10,
            ).generate(x)

            axs[int(i / 3), i % 3].imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=0)
        plt.show()
        return topics_dict
