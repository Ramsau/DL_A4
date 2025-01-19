import shap
import lime
import lime.lime_text
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from captum.attr import IntegratedGradients
from dataset import IMDBDataset
from model import RNNClassifier
from config import get_config
from vocab import Vocabulary

# Ensure required libraries are available
try:
    from IPython.display import display, HTML
except ImportError:
    import subprocess

    subprocess.check_call(["python", "-m", "pip", "install", "ipython"])
    from IPython.display import display, HTML


def tokenize(text, vocab):
    """
    Tokenizes input text and converts it into numerical form using the vocabulary.
    Handles different input types correctly (str, list, numpy array, etc.).
    """
    if isinstance(text, np.ndarray):  # SHAP sometimes passes arrays
        text = " ".join(text.astype(str).flatten())  # Convert full array to string

    if isinstance(text, list):  # If SHAP passes a list of words, join into string
        text = " ".join(text)

    tokens = vocab.numericalize(text)
    return tokens if tokens else [vocab.stoi['<unk>']]


def predict_proba(model, vocab, device):
    def predict(texts):
        tokenized_texts = [torch.tensor(tokenize(text, vocab), dtype=torch.long) for text in texts]
        lengths = torch.tensor([len(seq) for seq in tokenized_texts], dtype=torch.long)

        valid_indices = [i for i, length in enumerate(lengths) if length > 0]
        if not valid_indices:
            return np.array([[0.5, 0.5]] * len(texts))  # Neutral predictions for empty inputs

        padded_texts = pad_sequence([tokenized_texts[i] for i in valid_indices], batch_first=True,
                                    padding_value=vocab.stoi['<pad>']).to("cpu")  # Move to CPU

        lengths = lengths.to("cpu")  # Move to CPU

        model.to("cpu")  # Ensure the model is on CPU for LIME
        model.eval()
        with torch.no_grad():
            inputs = padded_texts
            valid_lengths = lengths
            outputs = model(inputs, valid_lengths)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Move to NumPy for LIME

        full_probs = np.full((len(texts), 2), 0.5)  # Default neutral probability
        for i, idx in enumerate(valid_indices):
            full_probs[idx] = [1 - probs[i], probs[i]]
        return full_probs

    return predict




def explain_with_lime(model, vocab, device, text_sample):
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Positive'])
    predictor = predict_proba(model, vocab, device)

    exp = explainer.explain_instance(text_sample, predictor, num_features=10)

    # Save explanation as an HTML file
    html_file = "lime_explanation.html"
    exp.save_to_file(html_file)

    print(f"LIME explanation saved to {html_file}. Open this file in a browser to view it.")


def explain_with_shap(model, vocab, device, dataset, num_samples=5):
    pass


def main_xai():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.serialization.add_safe_globals([Vocabulary])  # Allowlist Vocabulary class
    checkpoint = torch.load(config['load_model_name'] + ".pth", map_location=device, weights_only=False)
    saved_vocab = checkpoint['vocab']

    model = RNNClassifier(vocab_size=len(saved_vocab.itos), embed_dim=config['embed_dim'],
                          hidden_dim=config['hidden_dim'], bidirectional=config['bidirectional'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    dataset_test = IMDBDataset(train=False)
    sample_index = np.random.randint(0, len(dataset_test))
    text_sample, true_label = dataset_test[sample_index]

    #print("\nUsing LIME to explain prediction on a sample:")
    #explain_with_lime(model, saved_vocab, device, text_sample)

    print("\nUsing SHAP to analyze feature importance:")
    explain_with_shap(model, saved_vocab, device, dataset_test)


if __name__ == '__main__':
    main_xai()
