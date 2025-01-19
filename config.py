def get_config():
    return {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "embed_dim": 100,
        "hidden_dim": 128,
        "bidirectional": True,
        "num_epochs": 10,
        "print_batches_label_amount": 0,
        "print_batches_raw": False,
        # 0-12499 = neg; 12500-24999 = pos
        "print_batches_raw_indices": [0, 12499, 12500, 24999],
        "save_model_name": "rnn_classifier",
        "load_model_name": "rnn_classifier"
}