import logging
import lime
import lime.lime_text
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from dataset import IMDBDataset
from model import RNNClassifier
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tokenize(text, vocab):
    if isinstance(text, np.ndarray):
        text = " ".join(text.astype(str).flatten())

    if isinstance(text, list):
        text = " ".join(text)

    tokens = vocab.numericalize(text)
    return tokens if tokens else [vocab.stoi['<unk>']]


def predict_proba(model, vocab, device):
    def predict(texts):
        tokenized_texts = [torch.tensor(tokenize(text, vocab), dtype=torch.long).to(device) for text in texts]
        lengths = torch.tensor([len(seq) for seq in tokenized_texts], dtype=torch.long).to(device)

        valid_indices = [i for i, length in enumerate(lengths) if length > 0]
        if not valid_indices:
            return np.array([[0.5, 0.5]] * len(texts))

        padded_texts = pad_sequence([tokenized_texts[i] for i in valid_indices], batch_first=True,
                                    padding_value=vocab.stoi['<pad>'])

        model.eval()
        with torch.no_grad():
            outputs = model(padded_texts, lengths)
            probs = torch.sigmoid(outputs).cpu().numpy()

        full_probs = np.full((len(texts), 2), 0.5)
        for i, idx in enumerate(valid_indices):
            full_probs[idx] = [1 - probs[i], probs[i]]
        return full_probs

    return predict



def explain_with_lime(model, vocab, device, text_sample):
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Positive'])
    predictor = predict_proba(model, vocab, device)

    exp = explainer.explain_instance(text_sample, predictor, num_features=10)

    html_file = "lime_explanation.html"
    exp.save_to_file(html_file)

    logger.info(f"LIME explanation saved to {html_file}. Open this file in a browser to view it.")



def load_model(config, device):
    checkpoint = torch.load(config['load_model_name'] + ".pth", map_location=device, weights_only=False)
    saved_vocab = checkpoint['vocab']

    model = RNNClassifier(vocab_size=len(saved_vocab.itos), embed_dim=config['embed_dim'],
                          hidden_dim=config['hidden_dim'], bidirectional=config['bidirectional'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, saved_vocab

def main_xai():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, saved_vocab = load_model(config, device)

    model.eval()

    dataset_test = IMDBDataset(train=False)
    sample_index = np.random.randint(0, len(dataset_test))
    text_sample, true_label = dataset_test[sample_index]

    logger.info("\nUsing LIME to explain prediction on a sample:")
    explain_with_lime(model, saved_vocab, device, text_sample)



if __name__ == '__main__':
    main_xai()
