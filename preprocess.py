import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ============================================================
# Download required NLTK resources
# ============================================================

def download_nltk_resources():

    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4"
    }

    for path, resource in resources.items():

        try:
            nltk.data.find(path)

        except LookupError:

            print(f"Downloading NLTK resource: {resource}")

            nltk.download(resource)


download_nltk_resources()


# ============================================================
# Initialize NLP tools
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# Text Preprocessing Function
# ============================================================

def preprocess_text(text):

    # Convert text to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    processed_tokens = []

    for token in tokens:

        if token not in stop_words:

            lemma = lemmatizer.lemmatize(
                token
            )

            processed_tokens.append(
                lemma
            )

    return " ".join(
        processed_tokens
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_text = "How can I track my order?"

    processed_text = preprocess_text(
        test_text
    )

    print("\n==============================")
    print("NLP PREPROCESSING TEST")
    print("==============================")

    print(
        "Original Text:",
        test_text
    )

    print(
        "Processed Text:",
        processed_text
    )

    print("==============================\n")