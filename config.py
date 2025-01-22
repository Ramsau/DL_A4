def get_config():
    return {
        "batch_size": 64,
        "learning_rate": 0.001,
        "embed_dim": 200,
        "hidden_dim": 256,
        "bidirectional": True,
        "num_epochs": 6,
        "print_batches_label_amount": 0,
        "print_batches_raw": False,
        # 0-12499 = neg; 12500-24999 = pos
        "print_batches_raw_indices": [0, 12499, 12500, 24999],
        "save_model_name": "rnn_classifier",
        "load_model_name": "rnn_classifier" # todo change for test.py
}


def get_grid_configs():
    grid = []

    batch_sizes = [32, 64, 128]
    learning_rates = [1e-2, 1e-3, 1e-4]
    embed_dims = [100, 200]
    hidden_dims = [128, 256]
    num_layers_list = [1, 2]
    bidirectional_opts = [True]

    num_epochs = [6]

    import itertools
    for (bs, lr, ed, hd, nl, bi, ep) in itertools.product(
            batch_sizes, learning_rates, embed_dims, hidden_dims, num_layers_list, bidirectional_opts, num_epochs):
        config = {
            "batch_size": bs,
            "learning_rate": lr,
            "embed_dim": ed,
            "hidden_dim": hd,
            "num_layers": nl,
            "bidirectional": bi,
            "num_epochs": ep,
            "print_batches_label_amount": 0,
            "print_batches_raw": False,
            "print_batches_raw_indices": [0, 12499, 12500, 24999],
            "save_model_name": f"rnn_classifier_{bs}_{lr}_{ed}_{hd}_{nl}",
            "load_model_name": f"rnn_classifier_{bs}_{lr}_{ed}_{hd}_{nl}"
        }
        grid.append(config)

    return grid