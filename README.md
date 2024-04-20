## Movie Recommendation System


### Table of Contents

1. [Reference Links](#reference-links)
2. [Project Overview](#project-overview)
    - [Dataset](#dataset)
    - [Imports and Data Loading](#imports-and-data-loading)
    - [Data Preprocessing](#data-preprocessing)
    - [Text Processing](#text-processing)
        - [Abstract Syntax Trees](#abstract-syntax-trees)
        - [Data Normalize](#data-normalize)
            - [Stemming](#stemming)
            - [Lemmatization](#lemmatization)
            - [Why Lemmatization is Preferred](#why-lemmatization-is-preferred)
        - [Vectorization](#vectorization)
    - [Model Building](#model-building)
        - [Euclidean Distance](#euclidean-distance)
        - [Cosine Similarity](#cosine-similarity)
        - [Why Use Cosine Similarity in My Model?](#why-use-cosine-similarity-in-my-model)
    - [Saving Results](#saving-results)
    - [Graphs & Observations](#graphs-and-observations)
3. [Installation](#installation)
    - [React Native Deployment](#react-native-deployment)
    - [Python](#python)
4. [Installation Instructions](#installation-instructions)
    - [React Native Deployment](#react-native-deployment-1)
    - [Python](#python)
5. [Conclusion](#conclusion)
6. [connect](#connect)



### Reference Links

#### The following links are for Reference and documentation of the used modules and models.

| TOPICS               | LINKS                                                                                   |
| --------------------| ----------------------------------------------------------------------------------------|
| Kaggle              | [kaggle.com](https://www.kaggle.com/docs/datasets)                                      |
| Data Set [Kaggle]   | [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)     |
| Numpy               | [Numpy.org](https://numpy.org/devdocs/user/whatisnumpy.html)                             |
| Pandas              | [Pandas.org](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)            |
| Abstract Syntax Trees | [Python/AST](https://docs.python.org/3/library/ast.html)                                 |
| Lemmatization       | [Nltk/Lemmatization](https://www.nltk.org/api/nltk.stem.wordnet.html)                    |
| Vectorization       | [Scikit Learn/CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)  |
| Cosine Similarity   | [Scikit Learn/Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) |

---

### **Project Overview**

In this project, I've developed a movie recommendation system using machine learning techniques. Below, I'll describe the models utilized, the imports, their functions, and why they were chosen.

#### Dataset 

I used a dataset from Kaggle.com. The dataset I used is TMDB 5000 movies dataset. Check the [reference links](#reference-links) section to know about Features and more about the dataset.

### Imports and Data Loading

I start by importing the necessary libraries, including NumPy and Pandas for data manipulation. Then, I load the dataset containing movie information from `CSV` (Comma-separated values) files using Pandas.

There are two files :

`tmdb_5000_movies.csv`

`tmdb_5000_credits.csv`


### Data Preprocessing

After loading the data, I perform several preprocessing steps:

- Merging datasets: I merge two datasets containing movie information and credits.
- Removing Null Values: I remove rows with missing values in the dataset.So it future if we perform any operation it will not thow NaN or data missing errors.
- Extracting Features: I extract relevant features such as genres, keywords, cast, and direction from the dataset.As i want to recommend the movies based on movie content and overview.

### Text Processing

Text data such as genres, keywords, cast, and direction are processed using various techniques:

**Certainly! Let's delve deeper into each of these concepts:**

### Abstract Syntax Trees

The data in the data Set is in JSON format of string.To convert it into python literal or container display syntax we used AST.

**JSON (JavaScript Object Notation)** is a lightweight data interchange format commonly used for representing structured data. In the context of this project, JSON is used to store information such as movie genres, keywords, cast, and direction.

When data is stored in JSON format, it's typically represented as a string.

The **`ast.literal_eval()`** function in Python is used to safely evaluate a string containing a Python literal or container display syntax (such as lists, dictionaries, tuples, etc.) and return the corresponding Python object.

This function is particularly useful when you want to **evaluate strings that contain Python syntax** but want to ensure that it is safe to do so, as it only evaluates strings containing literals and does not evaluate any arbitrary Python code.

Here's an example of how `ast.literal_eval()` can be used:

```python
import ast

# Define a string containing a Python list literal
list_string = "[1, 2, 3, 4, 5]"

# Use ast.literal_eval() to evaluate the string and convert it to a Python list
result_list = ast.literal_eval(list_string)

print(result_list)
```

Output:
```
[1, 2, 3, 4, 5]
```

In this example, `ast.literal_eval()` safely evaluates the string `"[1, 2, 3, 4, 5]"` and returns the corresponding Python list `[1, 2, 3, 4, 5]`.

Similarly, you can use `ast.literal_eval()` with other Python literals and container display syntax, such as **dictionaries, tuples, and sets,** as long as the string represents a valid Python expression.

### Data Normalize

Lemmatization and stemming are both techniques used in **natural language processing (NLP)** to reduce words to their base or root form. While they serve a similar purpose, they differ in their approach and the results they produce.

### Stemming

  Stemming involves **removing prefixes and suffixes from words to obtain their root form**. It operates on the principle of chopping off word endings to achieve this. Stemming algorithms may use rules or heuristics to perform this operation, which can sometimes lead to over-stemming (where different words are reduced to the same stem) or under-stemming (where words are not reduced enough).

  For example:
  - "running" -> "run"
  - "cars" -> "car"
  - "better" -> "better" (No change as "better" is already its root form)

  ### Lemmatization

  Lemmatization, on the other hand, **aims to determine the lemma or base form of a word** by considering its context and meaning. Unlike stemming, lemmatization takes into account the part of speech of the word and uses lexical knowledge databases (such as WordNet) to accurately identify the lemma.

    For example:
    - "running" -> "run"
    - "cars" -> "car"
    - "better" -> "good" (Correctly identified as the lemma of "better")

### Why Lemmatization is Preferred

  In many cases, lemmatization is preferred over stemming because it **produces more accurate and meaningful results**. Since lemmatization considers the context and meaning of words, it can correctly identify the base form even when the word undergoes complex inflectional changes. This is especially important in tasks like text classification, sentiment analysis, and information retrieval, where the precise meaning of words matters.

  In the context of the movie recommendation system, lemmatization is preferred because it ensures that similar words are treated as the same entity, leading to better matching and more accurate recommendations. **For example, lemmatization can correctly identify that "running", "runs", and "ran" all share the same base form "run"**, ensuring that movies with similar themes or keywords are appropriately matched. This enhances the effectiveness of the recommendation system in capturing the semantic similarities between movies and improving the user experience.

  Here's an example of lemmatization in Python using the NLTK (Natural Language Toolkit) library:
  

    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    word = "running"
    lemmatized_word = lemmatizer.lemmatize(word, pos='v')

    print(lemmatized_word)
    
  This will output: 

    run

### Vectorization

Vectorization is the process of converting text data into numerical vectors that machine learning models can understand. CountVectorizer is a commonly used technique for vectorizing text data.

CountVectorizer converts a collection of text documents into a matrix of token counts, where each row represents a document and each column represents a unique word (or token) in the corpus. The value at each position in the matrix represents **the frequency of occurrence of that word in the document**.

Here's an example of how CountVectorizer works in Python:
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

This will output:
```
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
```

Each row in the output array corresponds to a document, and each column represents a word. The value at each position indicates the frequency of that word in the corresponding document.

### Model Building

I utilize cosine similarity to calculate the similarity between movies based on their features. The steps involved in the recommendation process include:

Euclidean distance and cosine similarity are both metrics used to measure the similarity between vectors, but they approach the concept of similarity in different ways and have different applications.

  ### Euclidean Distance

  Euclidean distance measures **the straight-line distance between two points in Euclidean space**. In the context of vectors, it calculates the geometric distance between two points represented by vectors. Euclidean distance is calculated using the formula:


  ![equation](/readme_assets/euclidean_distance.png)


  - Euclidean distance considers both the magnitude and direction of vectors.
  - It is sensitive to differences in magnitudes between vectors.
  - It is useful when the magnitude of the vectors is important and you want to measure their spatial distance.

  ### Cosine Similarity

  Cosine similarity measures **the cosine of the angle between two vectors in a multi-dimensional space**. It indicates the similarity in direction between the vectors, regardless of their magnitudes. Cosine similarity is calculated using the formula:


  ![equation](/readme_assets/cosine_similarity.png)


  - Cosine similarity only considers the direction of vectors, ignoring their magnitudes.
  - It is particularly useful when the magnitude of vectors is not important, and you want to focus on their orientation or semantic similarity.
  - Cosine similarity is widely used in text analysis, information retrieval, and recommendation systems.

### Why Use Cosine Similarity in My Model?

  In many natural language processing (NLP) tasks, such as text analysis and recommendation systems, the focus is often on the semantic similarity between documents or text samples.**In these cases, the magnitude of vectors (e.g., word frequency) may not be relevant, but the orientation or direction of vectors (e.g., word semantics) is crucial.**

  Cosine similarity is preferred in such scenarios because it captures the semantic similarity between documents by measuring the angle between their vector representations. It is robust to changes in magnitude and scale, making it suitable for comparing text samples of varying lengths and frequencies.

  In the context of my model, which is likely dealing with textual data such as movie descriptions or user preferences, cosine similarity is likely chosen because it focuses on the semantic similarity between movies. By measuring the similarity in direction between movie vectors, cosine similarity can effectively identify movies with similar themes, genres, or content, leading to more accurate recommendations.

 ### Saving Results

  Finally, I saved the movie list and recommendation data as CSV files for future reference. Additionally, I serialized and saved the recommendation data and similarity matrix using pickle for later use And also in json file for App data.

  This movie recommendation engine provides personalized movie recommendations based on user preferences and similarity between movies, making it a valuable tool for users seeking new movie suggestions.

## Graphs and Observations

- The dataset Genres based graph:
  
  ![Genres](/graphs/genres.png)

- Directer & Number of movies they directed :
  
  ![director](/graphs/directors.png)

- Similarity plotting between movies
    - *note: This is not generated by me , it is taken as refrence and to visualize how the recommendation will work.*
  
  ![similarity](/graphs/similatity.png)
  

---


## Installation

- ### React Native Deployment
| Steps                              | Commands                                                                                                 |
|------------------------------------|----------------------------------------------------------------------------------------------------------|
| Clone the repository              | `git clone https://github.com/jagadeeshm007/Movie_Recommendation_System.git`                              |
| Navigate to the app directory     | `cd ./Movie_Recommendation_Engine/GUI/Movie_Recommendation_Search_App`                                   |
| Create a `.env` file              | `echo EXPO_API_KEY=YOUR_API_KEY > .env`                                                                  |
| Install dependencies              | `npm install`                                                                                             |
| Start the application locally     | `npm start`                                                                                               |

- ### Python Testing
| Steps                              | Commands                                                                                                 |
|------------------------------------|----------------------------------------------------------------------------------------------------------|
| Execute the Python script          | `python Movie_Recommendation_System.py`                                                                  |



## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`EXPO_API_KEY`

## Installation Instructions

### [React Native] Deployment

To deploy this application, you require a **TMDB API**, as well as **Node.js** and **React Native**. You can obtain a free TMDB API from [here](https://developer.themoviedb.org/reference/intro/getting-started). Follow the React Native Documentation for Installation. The link to the documentation is [here](https://reactnative.dev/docs/environment-setup).

After obtaining the TMDB API and installing Node.js & React Native, follow these installation steps:

1. First, clone the repository using the `git clone` command in the terminal:

```bash
git clone https://github.com/jagadeeshm007/Movie_Recommendation_System.git
```

2. Navigate to the folder `GUI/Movie_Recommendation_Search_App` using the following command in the terminal:

```bash
cd ./Movie_Recommendation_Engine/GUI/Movie_Recommendation_Search_App
```

3. Now, create a file named `.env` in that directory.

**🛑🪧Note:** Replace `YOUR_API_KEY` with your TMDB API Key.

```bash
echo EXPO_API_KEY=YOUR_API_KEY > .env
```

4. Run `npm install` to download required dependencies.

```bash
npm install
```

5. Use the `npm start` command to start the application locally.

```bash
npm start
```

⚡The application will run locally at [`localhost:8081`](localhost:8081/). Use web or Expo Go to test the application.

### **⚠️NOTE:**
For complete documentation about my React Native application check my repository.

🔗[**`Movie_Recommendation_Search_App`**](https://github.com/jagadeeshm007/Movie_Recommendation_Search_App)

### 🔗Python

Additionally, you can try out the Python code in the GUI to test it using the `Movie_Recommendation_System.py` file in the GUI folder.


#### Conclusion

This movie recommendation system provides personalized movie recommendations based on user preferences and similarity between movies. It can be further optimized and integrated into various applications to enhance user experience.

---


## Connect

[![linkedin](https://raw.githubusercontent.com/dheereshagrwal/colored-icons/master/public/icons/linkedin/linkedin.svg)](https://www.linkedin.com/in/mandalajagadeesh/)
