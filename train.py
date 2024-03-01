from data import AudioDataModule
import pytorch_lightning as pl
from model import AudioTransformer
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import local_env
import yaml

# Path to your YAML file
config_file_path = local_env.config_path

# Reading the YAML file
with open(config_file_path, "r") as file:
    config = yaml.safe_load(file)['config']

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    model = AudioTransformer(
        vocab_size=config["vocab_size"],
        window_size=config["window_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        activation=config["activation"],
        norm_first=config["norm_first"],
        bias=config["bias"],
    )
    datamodule = AudioDataModule(
        local_env.data_dir,
        config["batch_size"],
        config["window_size"],
        config["stride"],
    )
    model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=1)
    wandb_logger = WandbLogger(project="smpl-tr", log_model="all")
    trainer = pl.Trainer(
        max_epochs=250, callbacks=[model_checkpoint], logger=wandb_logger
    )
    trainer.fit(model=model, datamodule=datamodule)
