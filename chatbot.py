import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text


class FAQChatbot:

    def __init__(self, faq_file="data/faqs.csv", threshold=0.20):

        self.faq_file = faq_file
        self.threshold = threshold

        self.df = None
        self.vectorizer = None
        self.faq_vectors = None

        self.load_data()
        self.train_model()

    # Load FAQ dataset
    def load_data(self):

        if not os.path.exists(self.faq_file):
            raise FileNotFoundError(
                f"FAQ file not found: {self.faq_file}"
            )

        self.df = pd.read_csv(self.faq_file)

        # Check columns
        if "question" not in self.df.columns or "answer" not in self.df.columns:
            raise ValueError(
                "CSV must contain 'question' and 'answer' columns."
            )

        # Remove empty rows
        self.df = self.df.dropna(
            subset=["question", "answer"]
        ).reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("FAQ dataset is empty.")

    # Create TF-IDF vectors
    def train_model(self):

        self.df["processed_question"] = (
            self.df["question"].apply(preprocess_text)
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )

        self.faq_vectors = self.vectorizer.fit_transform(
            self.df["processed_question"]
        )

    # Find best FAQ answer
    def get_response(self, user_question):

        if not user_question.strip():
            return {
                "answer": "Please enter a question.",
                "score": 0.0,
                "matched_question": None
            }

        # Preprocess user question
        processed_question = preprocess_text(
            user_question
        )

        # Convert user question to TF-IDF vector
        user_vector = self.vectorizer.transform(
            [processed_question]
        )

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(
            user_vector,
            self.faq_vectors
        )[0]

        # Find highest score
        best_index = similarity_scores.argmax()
        best_score = similarity_scores[best_index]

        matched_question = self.df.iloc[
            best_index
        ]["question"]

        # Check threshold
        if best_score < self.threshold:
            return {
                "answer": (
                    "Sorry 😔 I couldn't find a relevant "
                    "answer to your question. Please try "
                    "asking in a different way."
                ),
                "score": float(best_score),
                "matched_question": None
            }

        answer = self.df.iloc[
            best_index
        ]["answer"]

        return {
            "answer": answer,
            "score": float(best_score),
            "matched_question": matched_question
        }


# Test chatbot
if __name__ == "__main__":

    print("\n==============================")
    print("🤖 FAQ CHATBOT TEST")
    print("==============================")

    try:

        bot = FAQChatbot()

        question = "How can I track my order?"

        result = bot.get_response(question)

        print("\nQuestion:")
        print(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSimilarity Score:")
        print(round(result["score"], 2))

        print("\nMatched FAQ:")
        print(result["matched_question"])

        print("\n==============================")
        print("✅ Chatbot test completed!")
        print("==============================")

    except Exception as e:

        print("\n❌ Error:")
        print(e)