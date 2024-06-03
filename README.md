# Information Retrieval Project

An information retrieval system that uses advanced natural language processing techniques to retrieve relevant documents from a corpus based on a user's query.

## Datasets

The datasets used in this project are:

- [Wikir](https://ir-datasets.com/wikir.html#wikir/en1k)
- [Lotte](https://ir-datasets.com/lotte.html#lotte/science)

## Project Stages

The project consists of the following stages:

1. **Data Preprocessing**: The data is cleaned by removing stop words, punctuations, and single letters. Synonym mapping is performed to unify different terms with the same meaning. Finally, lemmatization is applied to reduce words to their base or root form.

2. **Indexing**: The cleaned data is then indexed using TF-IDF (Term Frequency-Inverse Document Frequency), a numerical statistic that reflects how important a word is to a document in a collection or corpus.

3. **Matching and Ranking**: The system uses cosine similarity to match and rank documents based on their relevance to the user's query.

## Additional Features

- **Word Embedding using Word2Vec**: A Word2Vec model is trained on the Wikir dataset to create custom word embeddings.

- **Word Embedding using SBERT**: A pre-trained model from Sentence Transformers called "allMini-LM-L6-V2" is used for indexing on the Lotte dataset.

- **Personalization**: The system takes into account the user's past queries and location to provide more accurate and personalized results.

- **Clustering and Topic Detection**: The system represents the documents in a plot, distinguishes different clusters, and assigns distinct topics to each cluster.

## User Interface

The user interface for the app is built using the Streamlit library.

## Running the Application

To initiate the application, you need to navigate to the Python directory and execute the `app.py` script. Here are the commands you need to run:

```bash
cd python
python -m streamlit run app.py
```

## Installation

The following packages are used in this project and can be installed using pip:

```bash
pip install transformers torch==2.2.2 spacy gensim roman re tqdm geocoder sentence-transformers streamlit wordcloud textblob nltk sklearn numpy pandas matplotlib
```
