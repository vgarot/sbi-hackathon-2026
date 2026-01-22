import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from teddy.Teddy import Teddy
from teddy.lightning.teddy_lightning import TeddyLightning

from teddy.lightning.datamodule import BDS_datamodule
from teddy.data.Alphabet import Alphabet
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import json
import argparse
from teddy.lightning.callbacks import LogGrad, LogLR, Precision
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

def main(args):
    json_file = args.config_file
    with open(json_file) as f:
        config = json.load(f)
    
    pl.seed_everything(config["seed"])

    alphabet = Alphabet(list(config["alphabet"]))
    teddy = Teddy(alphabet, **config["teddy"])

    # os.environ["TORCHINDUCTOR_COMPILE_WORKER_POOL_SIZE"] = "0"

    torch.set_float32_matmul_precision("high")

    teddy_lightning = TeddyLightning(teddy=teddy, **config["teddy_lightning"])

    teddy_lightning = torch.compile(teddy_lightning, dynamic=False)

    datamodule = BDS_datamodule(alphabet=alphabet, **config["datamodule"])

    logger = WandbLogger(**config["wandb_logger"], config=config)
    teddy_lightning.save_hyperparameters({"config_file":config})

    modelCheckpoint = ModelCheckpoint(every_n_train_steps=config["model_save_every_train_step"], filename="{epoch}_{step}", save_top_k=-1)
    modelCheckpoint_epochs = ModelCheckpoint(every_n_epochs=1, filename="{epoch}_{step}", save_top_k=-1)

    # early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    #     monitor="val_total_loss",
    #     patience=config["early_stopping_patience"],
    #     mode="min",
    # )

    
    callbacks = [
        LogGrad(), 
        LogLR(config["trainer"]["log_every_n_steps"]), 
        # Precision(), 
        modelCheckpoint,
        modelCheckpoint_epochs,
        # early_stopping
        ]
    
    # plugins = [
    #     SLURMEnvironment(
    #         requeue_signal=signal.SIGUSR1,  # Signal to requeue the job
    #         auto_requeue=config["with_auto_resubmit"],  # Automatically requeue the job
    #     )
    # ]

    # profiler = pl.profilers.PyTorchProfiler(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     export_to_chrome=True,
    #     warmup=10
    # )

    trainer = pl.Trainer(**config["trainer"], 
                         callbacks=callbacks, 
                         logger=logger, 
                        #  profiler=profiler,
                        #  plugins=plugins,
                         )
    

    trainer.fit(teddy_lightning, datamodule)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config_file", type=str, help="Path to the config file", required=True)
    args = parser.parse_args()
    main(args)
