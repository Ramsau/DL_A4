import argparse
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from dataset import IMDBDataset
from model import RNNClassifier
from config import get_config

def collate_fn(batch, vocab, max_seq_len=None):
    texts, labels = zip(*batch)
    numericalized = []

    for text in texts:
        tokens = vocab.numericalize(text)
        if max_seq_len:
            tokens = tokens[:max_seq_len]
        if len(tokens) == 0:
            tokens = [vocab.stoi['<unk>']]
        numericalized.append(torch.tensor(tokens, dtype=torch.long))

    lengths = torch.tensor([len(seq) for seq in numericalized], dtype=torch.long)
    padded = pad_sequence(numericalized, batch_first=True, padding_value=vocab.stoi['<pad>'])
    labels = torch.tensor(labels, dtype=torch.float)
    return padded, lengths, labels, texts


def main(config, sample_index=None):
    checkpoint = torch.load("best_model/rnn_classifier_64_0.001_200_256_1_4_best.pth", map_location="cpu")
    saved_vocab = checkpoint['vocab']

    model = RNNClassifier(vocab_size=len(saved_vocab.itos), embed_dim=config['embed_dim'],
                          hidden_dim=config['hidden_dim'], bidirectional=config['bidirectional'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)

    dataset_test = IMDBDataset(train=False)

    if sample_index is None:
        sample_index = random.randint(0, len(dataset_test) - 1)
    else:
        if sample_index < 0 or sample_index >= len(dataset_test):
            raise ValueError(f"Sample index must be between 0 and {len(dataset_test) - 1}.")

    raw_text, true_label = dataset_test[sample_index]

    batch = [(raw_text, true_label)]
    inputs, lengths, labels, texts = collate_fn(batch, saved_vocab, max_seq_len=1000)
    inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)

    print("=" * 80)
    print("Raw text of the selected test sample:")
    print(f"{texts[0][:200]}...")
    print("=" * 80)

    with torch.no_grad():
        outputs = model(inputs, lengths)
        probs = torch.sigmoid(outputs)
        prediction = (probs >= 0.5).float()

    print("Random Test Sample Prediction:")
    print(f"Predicted probability: {probs.item():.4f}")
    print(f"Predicted label: {int(prediction.item())}")
    print(f"Ground truth label: {int(labels.item())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("sample_index", nargs="?", type=int, default=None)
    args = parser.parse_args()

    config = get_config()
    main(config, sample_index=args.sample_index)
