"""Train The UNet for ACAR data – USE  1/4 Quadrants, full network depth"""

#from MCM_functions import *
from U_net import *
import numpy as np
from torch.utils.data import DataLoader
import LION.CTtools.ct_geometry as ct
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from datetime import datetime
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import shutil
import torch
import torch.nn as nn

# May have to restart kernel to change GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)

Count_Rate = 200_000_000
rate_M = Count_Rate // 1_000_000
N = 256  # Size of the input data (gets symmetrized to 512x512 in PairedACARDataLoader)
max_samples = 2_048
batch_size = 16
EPOCHS = 150
# learning_rate = 5e-5
learning_rate = 1e-3

# Paired dataset: returns (measurement, target) for each sample
paired_dataset = DeepCormack_dataloader(
    measurement_folder = f'/store/LION/gfbl2/DeepCormack_Data/{rate_M}M_Counts/TPMD_Measurement',
    target_folder = f'/store/LION/gfbl2/DeepCormack_Data/TPMD_Ideal',
    file_pattern = '*.txt',
    N = N,
    max_samples = max_samples                           # Only use the first N samples
)

# Calculate sizes
train_size = int(0.8 * len(paired_dataset))
val_size = int(0.1 * len(paired_dataset))
test_size = len(paired_dataset) - train_size - val_size  # Ensure all samples are used

# Split into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(
    paired_dataset, [train_size, val_size, test_size]
)

# Get indices, filenames of test set from random_split – either that or use same random seed for loading files each time…
test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else test_dataset
test_filenames = [paired_dataset.measurement_files[i] for i in test_indices]


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_params = FBPConvNet.default_parameters()
model = FBPConvNet(model_parameters=model_params).to(device)
criterion = torch.nn.MSELoss()
#lr=1e-3
#lr=5e-2
#lr=1e-2    # 20250820_120526
#lr=1e-5
#lr=5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print("––––––––––|–|––––––––––")
print("Model Training Initialized")
print("–––––––––––––––––––––––")
#f.write(f"Seed used for split: {seed}\n")
print(f"Count Rate: {rate_M}M")
print(f"Total Samples: {max_samples}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {EPOCHS}")
print("–––––––––––––––––––––––")
print("Optimizer Learning Rate:", learning_rate)
print("–––––––––––––––––––––––")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'/store/LION/gfbl2/DeepCormack_Models/UNet/{rate_M}M_({timestamp})'
os.makedirs(output_dir, exist_ok=True)


"""Training the ML Model"""
def save_tensorboard_loss_plot(timestamp, output_dir):
    """
    Save a plot of training and validation loss from TensorBoard logs.

    Args:
        timestamp (str): Timestamp string used in run directory naming.
        output_dir (str): Directory to save the plot.
    """
    run_dir = f'runs/fbpconvnet_{timestamp}'
    train_dir = os.path.join(run_dir, 'Training vs. Validation Loss_Training')
    val_dir = os.path.join(run_dir, 'Training vs. Validation Loss_Validation')

    # Check if event files exist
    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        print("TensorBoard event directories not found.")
        return

    train_acc = EventAccumulator(train_dir)
    train_acc.Reload()
    val_acc = EventAccumulator(val_dir)
    val_acc.Reload()

    # Use the first tag in each
    train_tag = train_acc.Tags()['scalars'][0]
    val_tag = val_acc.Tags()['scalars'][0]

    train_loss = train_acc.Scalars(train_tag)
    val_loss = val_acc.Scalars(val_tag)

    epochs = [x.step for x in train_loss]
    train_values = [x.value for x in train_loss]
    val_values = [x.value for x in val_loss]

    plt.figure(figsize=(8,5))
    #plt.plot(epochs, train_values, label='Training Loss')
    #plt.plot(epochs, val_values, label='Validation Loss')
    plt.plot(epochs[1:], train_values[1:], label='Training Loss')
    plt.plot(epochs[1:], val_values[1:], label='Training Loss')
    #plt.ylim(0, 1e-3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss from TensorBoard')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, f'loss_plot_{timestamp}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    #print(f"Saved TensorBoard loss plot to {plot_path}")


def add_slice_idx_channel(batch, device):
    """
    Add slice index as an extra channel to the input batch.

    Args:
        batch (Tensor): shape (B, 5, H, W)
        device: torch device

    Returns:
        Tensor: shape (B, 6, H, W)
    """
    B, C, H, W = batch.shape
    slice_idx = torch.arange(C, dtype=torch.float32, device=device).view(1, C, 1, 1)
    slice_idx = slice_idx.expand(B, C, H, W)
    batch_with_idx = torch.cat([batch, slice_idx], dim=1)
    return batch_with_idx


def expand_quadrant_to_full(img_quadrant):
    """
    Expand a quadrant (B, 1, H, W) to a full image (B, 1, 2H, 2W) by symmetry.

    Args:
        img_quadrant (Tensor): shape (B, 1, H, W)

    Returns:
        Tensor: shape (B, 1, 2H, 2W)
    """
    B, C, H, W = img_quadrant.shape
    # Mirror horizontally and vertically
    right = torch.flip(img_quadrant, dims=[3])
    bottom = torch.flip(img_quadrant, dims=[2])
    bottom_right = torch.flip(img_quadrant, dims=[2, 3])
    top = torch.cat([img_quadrant, right], dim=3)
    bottom = torch.cat([bottom, bottom_right], dim=3)
    full = torch.cat([top, bottom], dim=2)
    return full


def preprocess_batch(meas, device):
    """
    Preprocess measurement batch: (B, 5, 256, 256) -> (B*5, 1, 256, 256), add slice_idx as channel.

    Args:
        meas (Tensor): (B, 5, 256, 256)
        device: torch device

    Returns:
        Tensor: (B*5, 2, 256, 256)
    """
    B, C, H, W = meas.shape
    # Reshape to (B*C, 1, H, W)
    meas = meas.view(-1, 1, H, W)
    # Add slice_idx as channel
    slice_idxs = torch.arange(C, dtype=torch.float32, device=device).repeat(B)
    slice_idxs = slice_idxs.view(-1, 1, 1, 1).expand(-1, 1, H, W)
    meas = torch.cat([meas, slice_idxs], dim=1)
    return meas



def preprocess_target(targ):
    """
    Preprocess target batch: (B, 5, 256, 256) -> (B*5, 1, 256, 256)

    Args:
        targ (Tensor): (B, 5, 256, 256)

    Returns:
        Tensor: (B*5, 1, 256, 256)
    """
    B, C, H, W = targ.shape
    return targ.view(-1, 1, H, W)


def train_one_epoch(epoch_index, tb_writer):
    """
    Train the model for one epoch and return the true average loss over the entire epoch.

    Args:
        epoch_index (int): The current epoch index.
        tb_writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (meas, targ) in enumerate(train_loader):
        # meas, targ: (B, 5, 256, 256)
        meas = meas.float().to(device)
        targ = targ.float().to(device)

        # Preprocess: flatten batch and add slice_idx
        meas_proc = preprocess_batch(meas, device)  # (B*5, 2, 256, 256)
        targ_proc = preprocess_target(targ)         # (B*5, 1, 256, 256)

        optimizer.zero_grad()
        outputs = model(meas_proc)                  # (B*5, 1, 256, 256)

        # Expand quadrants to full images
        outputs_full = expand_quadrant_to_full(outputs)
        targ_full = expand_quadrant_to_full(targ_proc)

        loss = criterion(outputs_full, targ_full)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Optional: log every N batches
        log_interval = max(1, len(train_loader) // 5)
        if (i + 1) % log_interval == 0:
            #print(f'  batch {i + 1} loss: {loss.item():.3e}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', loss.item(), tb_x)

    avg_loss = total_loss / num_batches
    return avg_loss


# TensorBoard writer
os.makedirs(output_dir, exist_ok=True)
#timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/fbpconvnet_{timestamp}')
epoch = 0
min_epoch_for_best = int(0.5 * EPOCHS)
best_val_so_far = float('inf')
best_vloss = float('inf')
start_time = time.time()

train_losses = []
val_losses = []

# Write test_set file names to .txt in Fermi_Model directory
test_files_path = os.path.join(output_dir, f'test_files_{timestamp}.txt')
with open(test_files_path, 'w') as f:
    for fname in test_filenames:
        f.write(f"{fname}\n")
        



for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f'EPOCH {epoch + 1}:')
    avg_loss = train_one_epoch(epoch, writer)
    train_losses.append(avg_loss)  # Store training loss

    # Validation loop (if you have a validation_loader)
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (vmeas, vtarg) in enumerate(val_loader):  # Change to validation_loader if available
            vmeas = vmeas.float().to(device)
            vtarg = vtarg.float().to(device)
            vmeas_proc = preprocess_batch(vmeas, device)  # (B*5, 2, 256, 256)
            vtarg_proc = preprocess_target(vtarg)         # (B*5, 1, 256, 256)
            voutputs = model(vmeas_proc)
            voutputs_full = expand_quadrant_to_full(voutputs)
            vtarg_full = expand_quadrant_to_full(vtarg_proc)
            vloss = criterion(voutputs_full, vtarg_full)
            running_vloss += vloss.item()
    avg_vloss = running_vloss / (i + 1)
    val_losses.append(avg_vloss)  # Store validation loss
    print(f'LOSS train {avg_loss:.3e} valid {avg_vloss:.3e}\n')

    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch + 1)
    writer.flush()

    # Checkpoint logic with flexibility - save every 50 epochs with 5-epoch window
    num = 5
    checkpoint_epochs = np.arange(EPOCHS//num, EPOCHS, EPOCHS//num)  # Include EPOCHS if divisible by 50
    if (epoch + 1) in checkpoint_epochs:
        # Save checkpoint every 50 epochs
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_vloss': best_vloss,
            'timestamp': timestamp
        }
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1:03d}_{timestamp}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path} (epoch {epoch + 1})")

    # Best model logic
    if epoch >= min_epoch_for_best:
        if avg_vloss < best_val_so_far:
            best_val_so_far = avg_vloss
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_vloss': avg_vloss,
                'best_train_loss': avg_loss,
                'timestamp': timestamp
            }
            best_model_path = os.path.join(output_dir, f'best_model_{timestamp}.pt')
            torch.save(best_checkpoint, best_model_path)
            print(f"Best model updated: {best_model_path} (epoch {epoch + 1})")

    # Update loss log file after every epoch
    loss_log_path = os.path.join(output_dir, f'loss_log_{timestamp}.txt')
    with open(loss_log_path, 'w') as f:
        f.write("Epoch\tTrainingLoss\tValidationLoss\n")
        for e, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{e}\t{train_loss:.8f}\t{val_loss:.8f}\n")

    save_tensorboard_loss_plot(timestamp, output_dir)

    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    epochs_left = EPOCHS - (epoch + 1)
    est_remaining = epoch_time * epochs_left

    print(f"Epoch time: {epoch_time/60:.2f} min")
    print(f"Elapsed: {elapsed/60:.2f} min")
    print(f"Estimated time remaining: {est_remaining/60:.2f} min")
    print("–––––––––––––––––––––––")

    # Move or copy the file
    shutil.copy('output_unet.log', os.path.join(output_dir, 'output_unet.log'))

    #epoch += 1
    

"""
# Save losses to a text file
loss_log_path = os.path.join('Fermi_Model', f'loss_log_{timestamp}.txt')
with open(loss_log_patah, 'w') as f:
    f.write("Epoch\tTrainingLoss\tValidationLoss\n")
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
        f.write(f"{epoch}\t{train_loss:.8f}\t{val_loss:.8f}\n")
"""

final_model_path = os.path.join(output_dir, f'final_model_{timestamp}.pt')
torch.save(model.state_dict(), final_model_path)
print("=======================")
print(f"Loss log saved to {loss_log_path}", flush=True)
print(f"Final model saved to {final_model_path}", flush=True)
print("Training complete.", flush=True)
print("=======================")


# Move or copy the file
#shutil.copy('output.log', os.path.join(output_dir, 'output.log'))
