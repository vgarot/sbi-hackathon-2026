import lightning.pytorch as pl
from teddy.networks.embedding.Teddy import Teddy
from teddy.networks.inference.RegressionLayer import RegressionLayer
from teddy.losses.QuantileLoss import QuantileLoss
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import Softplus


class TeddyLightning(pl.LightningModule):
    def __init__(self,
                 teddy:Teddy, 
                 learning_rate:float = 2e-4,
                 warmup_steps:int = 1000,
                 normalization:dict= {
                     "R_0": [1,5],
                     "dur": [0.1,1]
                 },
                 scale_dur_loss:float = 1.0,
                 weight_decay:float = 0.01,
                 betas:tuple = (0.9, 0.98),
                 scheduler_type:str = "linear",
                 normalized_loss:bool = False,
                 inference_network = "regression",
                 inference_network_kwargs:dict = {
                        "output_dim":2,
                        "hidden_dim":128,
                 },
                 ):
        super().__init__()
        
        self.teddy = teddy

        embed_dim = teddy.embedding.embed_dim
        if inference_network == "regression":
            output_dim = inference_network_kwargs["output_dim"]
            hidden_dim = inference_network_kwargs["hidden_dim"]
            self.inference_network = RegressionLayer(embed_dim,output_dim*3,
            hidden_dim)
        else:
            raise NotImplementedError(f"Inference network {inference_network} not implemented.")

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.normalization = normalization

        self.normalized_loss = normalized_loss
        self.LowerCrIntLoss = QuantileLoss(alpha=0.025, normalized=self.normalized_loss)
        self.UpperCrIntLoss = QuantileLoss(alpha=0.975, normalized=self.normalized_loss)
        self.MedianLoss = QuantileLoss(alpha=0.5, normalized=self.normalized_loss)

        self.scale_dur_loss = scale_dur_loss
        self.weight_decay = weight_decay
        self.betas = betas
        self.scheduler_type = scheduler_type


        self.save_hyperparameters()

    def forward(self, input):

        embedding = self.teddy.forward(input)

        output = self.regression(embedding)

        return output
    
    def _normalize(self, tensor, key):
        min,max = self.normalization[key]
        return (tensor - min) / (max - min)
    
    def _renormalize(self, tensor, key):
        min,max = self.normalization[key]
        return tensor * (max - min) + min

    def _step(self, batch, batch_idx,mode):
        input, label = batch
        output = self.forward(input)

        R_0_pred = self._normalize(output[:,0:3],"R_0")
        R_0_true = self._normalize(label[:,0],"R_0")
        dur_pred = self._normalize(output[:,3:6],"dur")
        dur_true = self._normalize(label[:,1],"dur")

        R_0_LowerCrIntLoss = self.LowerCrIntLoss(R_0_pred[:,0],R_0_true)
        R_0_UpperCrIntLoss = self.UpperCrIntLoss(R_0_pred[:,2],R_0_true)
        R_0_MedianLoss = self.MedianLoss(R_0_pred[:,1],R_0_true)

        dur_LowerCrIntLoss = self.LowerCrIntLoss(dur_pred[:,0],dur_true)
        dur_UpperCrIntLoss = self.UpperCrIntLoss(dur_pred[:,2],dur_true)
        dur_MedianLoss = self.MedianLoss(dur_pred[:,1],dur_true)

        R_0_loss = R_0_LowerCrIntLoss + R_0_UpperCrIntLoss + R_0_MedianLoss
        dur_loss = dur_LowerCrIntLoss + dur_UpperCrIntLoss + dur_MedianLoss
        total_loss = R_0_loss + self.scale_dur_loss * dur_loss #it is harder to predict the duration for some reasons
        self.log_dict(
            {
                f"{mode}_R_0_LowerCrIntLoss": R_0_LowerCrIntLoss,
                f"{mode}_R_0_UpperCrIntLoss": R_0_UpperCrIntLoss,
                f"{mode}_R_0_MedianLoss": R_0_MedianLoss,
                f"{mode}_dur_LowerCrIntLoss": dur_LowerCrIntLoss,
                f"{mode}_dur_UpperCrIntLoss": dur_UpperCrIntLoss,
                f"{mode}_dur_MedianLoss": dur_MedianLoss,
                f"{mode}_R_0_loss": R_0_loss,
                f"{mode}_dur_loss": dur_loss,
                f"{mode}_total_loss": total_loss,
            },
            sync_dist=True,
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def predict_step(self, batch, batch_idx):
        name,input,s  = batch
        output = self.forward(input)

        return (name, output, input[1], s) # return name, predictions, and shapes

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.learning_rate,
                          weight_decay=self.weight_decay,
                          betas=self.betas,
                          )
        if self.hparams.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.max_steps,
            )
        else:  # default to linear scheduler
            assert self.hparams.scheduler_type == "linear", "Scheduler type must be either 'cosine' or 'linear'."

            scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class Teddy_NPE(pl.LightningModule):
    def __init__(self,
                 teddy:Teddy, 
                 npe_network,
                 learning_rate:float = 2e-4,
                 warmup_steps:int = 1000,
                 ):
        super().__init__()
        
        self.teddy = teddy
        self.npe_network = npe_network

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        self.save_hyperparameters()

    def forward(self, input):

        embedding = self.teddy.forward(input)

        output = self.npe_network.forward(
            embedding
            )

        return output

    def _step(self, batch, batch_idx,mode):
        input, label = batch
        output = self.forward(input)

        loss = self.npe_network.loss(output,label)

        self.log_dict(
            {
                f"{mode}_loss": loss,
            },
            sync_dist=True,
        )
        return loss