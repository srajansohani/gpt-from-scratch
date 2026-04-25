

GPT_CONFIG_124M = {
    "vacab_size": 50257,  ##Vocab Size
    "context_length": 1024, ##No fo tokens we will be using in once pass 
    "emb_dim": 768, ##Dimension of the embedding vector for each token
    "num_heads": 12, ##Number of attention heads
    "n_layers": 12, ##Number of transformer blocks (or layers) in the model
    "drop_rate": 0.1, ##Dropout rate for regularization
    "qkv_bias": False ##Whether to include bias terms in the query, key, and value projections in the attention mechanism
}