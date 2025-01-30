import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, log_loss
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
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
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

train_probs = model.predict_proba(texts_train)
val_probs = model.predict_proba(texts_val)
test_probs = model.predict_proba(texts_test)

train_logloss = log_loss(labels_train, train_probs)
val_logloss = log_loss(labels_val, val_probs)
test_logloss = log_loss(labels_test, test_probs)

print(f"Train Log Loss: {train_logloss:.4f}")
print(f"Val Log Loss:   {val_logloss:.4f}")
print(f"Test Log Loss:  {test_logloss:.4f}")

print("Saving model...")
with open("tfidf_logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully.")

print("Plotting confusion matrix for Test set...")
cm_test = confusion_matrix(labels_test, test_preds)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                   display_labels=["Negative", "Positive"])
disp_test.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (Test Set)")
plt.show()

print("Plotting learning curves...")

X_train_all = texts_train
y_train_all = labels_train

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_train_all,
    y_train_all,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

plt.title("Learning Curve (LogisticRegression)")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.grid(True)
plt.show()
