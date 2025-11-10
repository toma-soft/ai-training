class GPTConfig:
    def __init__(
        self,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        d_model=256,
        d_ff=1024,
        block_size=128,
        dropout=0.0,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.block_size = block_size
        self.dropout = dropout
