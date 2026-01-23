from teddy.lightning.datamodule import BDS_datamodule
import teddy.data.Alphabet as alphabet
from copy import deepcopy



import torch
from torch.optim import Adam
import gc

from sbi.analysis import pairplot
import matplotlib.pyplot as plt

from sbi.neural_nets.net_builders import build_nsf
from sbi.neural_nets.embedding_nets import CausalCNNEmbedding
import numpy as np
from teddy.networks.embedding.Teddy import Teddy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    alphabet_instance = alphabet.Alphabet(list( "ATGCX-"))

    train_ratio = 0.8
    batch_size = 256
    val_batch_size = batch_size

    # Optimize dataloader performance
    num_workers = 8  # Parallel data loading workers

    #msa = MsaLabels(dir = "data/example/seq", alphabet=alphabet_instance, limit_size=200)
    data = BDS_datamodule(
                        data_dir = "/lustre/fsn1/projects/rech/xzo/uhd27ew/sbi_test/seq/", 
                        alphabet=alphabet_instance, 
                        limit_size=200,
                        max_sites_len=200,
                        train_ratio=train_ratio, 
                        val_batch_size=val_batch_size, 
                        batch_size=batch_size,
                        num_workers=num_workers,
                        prefetch_factor=1,
                        persistent_workers=False, 
                        pin_memory=False, 
                        cache_dir = "/lustre/fsn1/projects/rech/xzo/uhd27ew/sbi_test/cache",
                        flatten_mode = False,
                        )

    data.setup()
    # Density estimator with first sequence and prior just for the dimensionality
    torch.manual_seed(0)

    # Define training params
    learning_rate = 5e-4
    validation_fraction = 0.1  # 10% of the data will be used for validation
    stop_after_epochs = 5  # Stop training after 5 epochs with no improvement
    max_num_epochs = 2**31 - 1

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    train_iter = iter(train_loader)
    batch = next(train_iter)

    x_batch = batch[0].to(device)
    theta_batch = batch[1].to(device)

    print(f"Initial batch shapes - x: {x_batch.shape}, theta: {theta_batch.shape}")

    # embedding_net = CausalCNNEmbedding(  # SOLUTION
    #     input_shape=(batch[0][0].shape[1],),      # 20000 timepoints
    #     in_channels=1,            # 1 channel: sequence data
    #     output_dim=20,            # Compress to 20 learned summary features
    #     num_conv_layers=5,        # Number of dilated causal conv layers
    #     kernel_size=2,            # Kernel size for convolutions
    # ).to(device)

    embedding_net = Teddy(
        alphabet=alphabet_instance,
        embed_dim=64,
        nb_heads=4,
        ffn_dim=128,
        nb_layers=3,
    ).to(device)

    density_estimator = build_nsf(theta_batch, x_batch, z_score_y="none", z_score_x="none", embedding_net=embedding_net).to(device) # theta batch dimension, x batch dimension

    optimizer = Adam(list(density_estimator.parameters()), lr=learning_rate)

    # Clean up initialization batch
    # del train_iter, batch, x_batch, theta_batch
    gc.collect()

    # === OPTIMIZED TRAINING LOOP ===
    epoch = 0
    best_val_loss = float("Inf")
    epochs_since_last_improvement = 0
    converged = False

    while epoch <= max_num_epochs and not converged:
        # === TRAINING PHASE ===
        density_estimator.train()
        train_loss_sum = 0
        num_train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Convert batch data to tensors
            # x_batch = batch[0][0].squeeze()
            # theta_batch = torch.as_tensor(batch[1], dtype=torch.float32).squeeze()

            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)

            #print(f"Training Batch {batch_idx}: x shape {x_batch.shape}, theta shape {theta_batch.shape}")

            # Forward pass and loss computation
            optimizer.zero_grad()
            train_losses = density_estimator.loss(theta_batch, x_batch)
            train_loss = torch.mean(train_losses)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss_sum += train_losses.sum().item()
            num_train_samples += theta_batch.size(0)
            
            # CRITICAL: Immediate cleanup after each batch
            del x_batch, theta_batch, train_losses, train_loss
            
            # Periodic garbage collection during training
            if batch_idx % 10 == 0:
                gc.collect()

        epoch += 1
        train_loss_average = train_loss_sum / num_train_samples

        # === VALIDATION PHASE ===
        density_estimator.eval()
        val_loss_sum = 0
        num_val_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x_batch = batch[0].to(device)
                theta_batch = batch[1].to(device)
                
                val_losses = density_estimator.loss(theta_batch, x_batch)
                val_loss_sum += val_losses.sum().item()
                num_val_samples += theta_batch.size(0)
                
                # Immediate cleanup
                del x_batch, theta_batch, val_losses
                
                if batch_idx % 5 == 0:
                    gc.collect()
        
        val_loss = val_loss_sum / num_val_samples

        # === MODEL CHECKPOINTING ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_last_improvement = 0
            # Store only state dict, not entire model
            best_model_state_dict = deepcopy(density_estimator.state_dict())
        else:
            epochs_since_last_improvement += 1

        # === CONVERGENCE CHECK ===
        if epochs_since_last_improvement > stop_after_epochs - 1:
            density_estimator.load_state_dict(best_model_state_dict)
            converged = True
            print(f'\nNeural network successfully converged after {epoch} epochs')
        else:
            print(f"Epoch {epoch}: Train loss: {train_loss_average:.4f}, Val loss: {val_loss:.4f}, Stops in: {stop_after_epochs - epochs_since_last_improvement}'")
        
        # Force garbage collection after each epoch
        gc.collect()

    # === POST-TRAINING CLEANUP ===
    print("\nCleaning up training resources...")

    # Delete dataloaders to free worker memory
    del val_loader
    gc.collect()

    # Optional: If using CUDA, clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training complete and memory cleaned!")



    # TESTING WITH TRAINING DATA

    train_iter = iter(train_loader)
    batch = next(train_iter)

    x_o = batch[0]
    theta_o = batch[1]

    print(f"Shape of x_o: {x_o.shape}            # Must have a batch dimension")

    samples = density_estimator.sample((10000,), condition=x_o.to(device)).detach()
    print(
        f"Shape of samples: {samples.shape}  # Samples are returned with a batch dimension."
    )

    samples = samples.squeeze(dim=1)
    print(f"Shape of samples: {samples.shape}     # Removed batch dimension.")
    
    # Extract only the parameter dimensions (first 2 dimensions correspond to R_0 and delta)
    samples = samples[:, :2]
    print(f"Shape of samples after extracting parameters: {samples.shape}")

    # Visualize posterior with pairplot
    param_labels = [r"$R_0$", r"$\delta$"]

    fig, axes = pairplot(
        samples.cpu(),
        #limits=[[0.05, 0.15], [0.01, 0.03], [0.005, 0.03], [0.005, 0.15]],
        labels=param_labels,
        figsize=(8, 8),
        points=theta_o,  # True parameters
        points_colors="r",
    )
    plt.suptitle("NPE Posterior (sbi)", y=1.02)
    plt.savefig("npe_posterior_training_data.png", bbox_inches="tight")



    return








if __name__ == "__main__":
    
    main()