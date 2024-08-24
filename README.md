# language_detection_model


## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Text Preprocessing](#text-preprocessing)
6. [Modeling](#modeling)
7. [Evaluation](#evaluation)
8. [Visualization](#visualization)
9. [Contributing](#contributing)


## Introduction

This project implements a Language Detection Model that classifies text into one of 17 languages. Using machine learning algorithms and Natural Language Processing (NLP) techniques, this model achieves high accuracy. The Support Vector Classification (SVC) model performed best with an accuracy of **98.3**. It can detect 


1)English

2) Malayalam

3) Hindi


4) Tamil

5) Kannada


6) French

7) Spanish

8) Portuguese

9) Italian

10) Russian

11) Sweedish

12) Dutch

13) Arabic

14) Turkish

15) German

16) Danish

17) Greek

## Features

- **Language Detection:** Classify text into 17 different languages.
- **NLP Techniques:** Perform punctuation removal and vectorization for text preprocessing.
- **Modeling:** Apply various machine learning algorithms for effective language classification.
- **Evaluation:** Measure and compare model performance with accuracy metrics.

## Installation

To set up the environment and install the required libraries, clone the repository and use the following commands:

```bash
git clone https://github.com/yourusername/language-detection-model.git
cd language-detection-model
pip install -r requirements.txt
```

### Dependencies

- Python 3.x
- pandas
- scikit-learn
- xgboost
- nltk (for text preprocessing)

Create a `requirements.txt` file with the following content (adjust versions as necessary):

```
pandas==1.3.3
scikit-learn==0.24.2
xgboost==1.4.2
nltk==3.6.3
```

## Usage

1. **Prepare Your Data:** Ensure your text data is formatted correctly.
2. **Run Preprocessing:** Execute the preprocessing script to remove punctuation and vectorize the text data.
3. **Train Models:** Use the provided scripts to train various machine learning models.
4. **Evaluate Models:** Assess the models’ performance and compare their accuracy scores.
5. **Make Predictions:** Use the trained model to predict the language of new text inputs.

Example code for making predictions:

```python
from model import LanguageDetector  # Adjust import based on your module structure

# Load the trained model
model = LanguageDetector.load('path/to/model.pkl')

# Example text for prediction
text = "This is a test sentence."

# Predict the language
language = model.predict(text)
print(f"The detected language is: {language}")
```

## Text Preprocessing

The preprocessing steps include:

- **Removing Punctuation:** Eliminate unnecessary punctuation marks from the text.
- **Vectorization:** Convert the cleaned text data into numerical features using TF-IDF vectorization.

## Modeling

The following machine learning algorithms were applied for language classification:

- **Support Vector Classification (SVC)**
- **XGBoost**
- **Decision Tree**
- **Random Forest**

TF-IDF vectorization was used to transform text data into numerical features suitable for these algorithms.

## Evaluation

Model performance was evaluated based on accuracy scores. The SVC model achieved the highest accuracy of **0.983**, demonstrating its effectiveness in language detection.

## Visualization

- **Decision Tree Plot:** Visualize the structure of the Decision Tree model to understand its decision-making process.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For significant changes, open an issue to discuss the proposed modifications before submitting.



Feel free to adjust any sections further if needed!
