import shap
import lime
import lime.lime_text
import torch
import numpy as np
from dataset import IMDBDataset
from model import RNNClassifier

checkpoint = torch.load("rnn_classifier.pth", map_location="cpu")
vocab = checkpoint['vocab']
model = RNNClassifier(vocab_size=len(vocab.itos), embed_dim=100, hidden_dim=128, bidirectional=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict_proba(texts):
    processed_texts = [torch.tensor(vocab.numericalize(text), dtype=torch.long) for text in texts]
    lengths = torch.tensor([len(seq) for seq in processed_texts], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(processed_texts, batch_first=True, padding_value=vocab.stoi['<pad>'])

    with torch.no_grad():
        outputs = model(padded, lengths)
        probs = torch.sigmoid(outputs).numpy()
        return np.vstack([1 - probs, probs]).T

dataset_test = IMDBDataset(train=False)
texts_test, labels_test = zip(*dataset_test[:100])

explainer_shap = shap.Explainer(predict_proba, texts_test)
shap_values = explainer_shap(texts_test[:10])  #first 10 samples

shap.text_plot(shap_values)

explainer_lime = lime.lime_text.LimeTextExplainer(class_names=["Negative", "Positive"])

sample_text = texts_test[0]
exp = explainer_lime.explain_instance(sample_text, predict_proba, num_features=10)
exp.show_in_notebook(text=True)
