# data preprocessing

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logger

nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_dataframe(df, col='text'):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.
    """
    logger.info("Starting text preprocessing for column: '%s'", col)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = ''.join([char for char in text if not char.isdigit()])  # Remove digits
        text = text.lower()  # Lowercase
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()  # Extra spaces
        text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
        return text

    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])  # Drop empty rows
    logger.info("Text preprocessing completed for column: '%s'", col)
    return df

def main():
    try:
        logger.info("==== Data Preprocessing Started ====")

        # Load data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info('Raw train and test data loaded successfully.')

        # Process data
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        # Save processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.info('Processed train and test data saved to %s', data_path)
        logger.info("==== Data Preprocessing Completed Successfully ====")

    except Exception as e:
        logger.error('❌ Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
