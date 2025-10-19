# Sentiment Analysis of Amazon Alexa Reviews using NLP and Machine Learning
A Natural Language Processing (NLP) and Machine Learning project that analyzes 10,000+ Amazon Echo product reviews to classify customer sentiments as positive, negative, or neutral. Includes data preprocessing, feature extraction (TF-IDF &amp; Bag of Words), model training, and performance evaluation with visualization.

# Sentiment Analysis of Amazon Alexa Reviews

This project performs sentiment analysis on Amazon Alexa reviews using Natural Language Processing (NLP) and Machine Learning. The goal is to classify customer reviews as either positive (1) or negative (0) based on the text of the review.

This repository contains the Jupyter Notebook with the complete analysis, the dataset used, and the final trained models.

## Project Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading & Exploration (EDA):**
    * The dataset (`amazon_alexa.tsv`) is loaded into a pandas DataFrame.
    * Exploratory Data Analysis is performed to understand the data, check for null values, and analyze the distribution of ratings and feedback.
    * Visualizations (bar charts and pie charts) are used to show the distribution of:
        * **Ratings:** The majority of reviews are 5-star.
        * **Feedback:** The dataset is imbalanced, with ~91.9% positive feedback (1) and ~8.1% negative feedback (0).
        * **Product Variations:** "Black Dot" and "Charcoal Fabric" are the most common product variations.

2.  **Data Preprocessing (NLP):**
    * A new feature, `length`, is created to store the length of each review.
    * The `verified_reviews` text is cleaned by:
        * Removing punctuation and special characters.
        * Converting all text to lowercase.
        * Removing common English stopwords (e.g., "the", "is", "a").
        * Applying stemming using `PorterStemmer` from NLTK to reduce words to their root form (e.g., "loved", "loving" -> "love").

3.  **Feature Engineering (Vectorization):**
    * The cleaned text corpus is converted into a numerical format using the **Bag-of-Words (BoW)** model.
    * `CountVectorizer` from scikit-learn is used to create a sparse matrix of token counts.

4.  **Model Training:**
    * The data is split into training (80%) and testing (20%) sets.
    * Two different classification models are trained on the BoW features to predict the `feedback` label:
        1.  **Random Forest Classifier**
        2.  **XGBoost Classifier**

5.  **Model Evaluation:**
    * The performance of both models is evaluated on the test set using:
        * **Accuracy Score**
        * **Confusion Matrix**

6.  **Model Saving:**
    * The trained models (`model_rf.pkl`, `model_xgb.pkl`), the `CountVectorizer` (`countVectorizer.pkl`), and the `MinMaxScaler` (`scaler.pkl`) are saved as pickle files for future use in production or for making new predictions without retraining.

## Dataset

* **`amazon_alexa.tsv`**: A tab-separated file containing 3150 reviews.
* **Columns:**
    * `rating`: The star rating given by the user (1-5).
    * `date`: The date of the review.
    * `variation`: The specific model/color of the Alexa product.
    * `verified_reviews`: The raw text of the review.
    * `feedback`: The target label (1 for positive, 0 for negative).

## Files in This Repository

* **`Sentiment_Analysis.ipynb`**: The main Jupyter Notebook containing all Python code for data loading, preprocessing, training, and evaluation.
* **`amazon_alexa.tsv`**: The raw dataset used for training and testing.
* **`countVectorizer.pkl`**: A saved pickle file for the `CountVectorizer` object, used to transform new text data.
* **`model_rf.pkl`**: The saved `RandomForestClassifier` model.
* **`model_xgb.pkl`**: The saved `XGBClassifier` model.
* **`scaler.pkl`**: The saved `MinMaxScaler` object.
* **`SentimentBulk.csv`**: A sample CSV file showing sentences that can be used for bulk prediction.
* **`Predictions.csv`**: A sample CSV file showing the predictions made by the model on the bulk sentiment file.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not provided, you can install the necessary libraries manually:
    ```bash
    pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost wordcloud
    ```
    You will also need to download the NLTK stopwords package:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

3.  **Run the Notebook:**
    Launch Jupyter Notebook and open `Sentiment_Analysis.ipynb` to see the full analysis.
    ```bash
    jupyter notebook Sentiment_Analysis.ipynb
    ```

4.  **Making New Predictions:**
    You can load the saved `.pkl` models and the vectorizer to predict the sentiment of new, unseen reviews.
