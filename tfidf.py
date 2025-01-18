import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataset import IMDBDataset
from vocab import tokenize

print("Loading dataset...")
dataset = IMDBDataset(train=True)
dataset_test = IMDBDataset(train=False)

texts_train, labels_train = zip(*dataset)
texts_test, labels_test = zip(*dataset_test)

texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_train, labels_train, test_size=0.2, random_state=42
)

def preprocess_texts(texts):
    return [" ".join(tokenize(text)) for text in texts]

print("Preprocessing texts...")
texts_train = preprocess_texts(texts_train)
texts_val = preprocess_texts(texts_val)
texts_test = preprocess_texts(texts_test)

print("Creating model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),  #vocab size  5000
    ('log_reg', LogisticRegression(max_iter=1000))
])

print("Training model...")
model.fit(texts_train, labels_train)

val_preds = model.predict(texts_val)
val_accuracy = accuracy_score(labels_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

test_preds = model.predict(texts_test)
test_accuracy = accuracy_score(labels_test, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Saving model...")
with open("tfidf_logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully.")