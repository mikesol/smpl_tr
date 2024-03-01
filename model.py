import torch
from torch import optim, nn
import pytorch_lightning as L


class AudioTransformer(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        window_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        activation,
        norm_first,
        bias,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_embedding_bit_depth = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.decoder_embedding_bit_depth = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.encoder_embedding_time = torch.nn.Embedding(
            num_embeddings=window_size, embedding_dim=d_model
        )
        self.decoder_embedding_time = torch.nn.Embedding(
            num_embeddings=window_size, embedding_dim=d_model
        )
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=norm_first,
            bias=bias,
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.dense_final = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, ipt, tgt, rg, mask):
        # sometimes, mask has batch because of how the dataloader works
        # if so, nix it
        mask_shape = mask.shape
        if len(mask_shape) == 3:
            mask = mask[0]
        ipt = self.encoder_embedding_bit_depth(ipt) + self.encoder_embedding_time(rg)
        tgt = self.decoder_embedding_bit_depth(tgt) + self.decoder_embedding_time(rg)
        o = self.transformer(ipt, tgt, src_mask=mask, tgt_mask=mask)
        o = self.ln_final(o)
        o = self.dense_final(o)
        return o

    def training_step(self, batch):
        x0, x1, y, rg, m = batch
        x_hat = self(x0, x1, rg, m)
        print('x_hat', x_hat.shape, x_hat.dtype, y.shape, y.dtype)
        loss = nn.CrossEntropyLoss(x_hat, y)
        return loss

    def validation_step(self, batch):
        x0, x1, y, rg, m = batch
        x_hat = self(x0, x1, rg, m)
        print('x_hat', x_hat.shape, x_hat.dtype, y.shape, y.dtype)
        loss = nn.CrossEntropyLoss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 16
    window_size = 16
    model = AudioTransformer(
        vocab_size=2**16,
        window_size=window_size,
        d_model=512,
        nhead=16,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="gelu",
        norm_first=True,
        bias=False,
    )
    summary(
        model,
        input_data=[
            torch.ones((batch_size, window_size), dtype=torch.int32),
            torch.ones((batch_size, window_size), dtype=torch.int32),
            torch.ones((batch_size, window_size), dtype=torch.int32),
            nn.Transformer.generate_square_subsequent_mask(window_size),
        ],
    )
