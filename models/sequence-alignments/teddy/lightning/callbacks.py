from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
import torch


class LogGrad(Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.logger.watch(pl_module,log='all', log_graph=False)
        print("Watching model!")

class LogLR(Callback):
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_step % self.log_every_n_steps == 0:
            # Log the lr if the rank is 0
            if trainer.is_global_zero:
                # Log the learning rate
                trainer.logger.experiment.log({"lr": trainer.optimizers[0].param_groups[0]['lr']})
    
    
class Precision(Callback):
    def setup(self, trainer, pl_module, stage: str) -> None:
        torch.set_float32_matmul_precision('medium')