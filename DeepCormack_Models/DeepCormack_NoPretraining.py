"""
DeepCormack_NoPretraining.py
Train a pipeline: RDF (via MLP) → reconstructed image → UNet → loss vs. ground truth.
"""
import os
import re
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from DeepCormack_MLP import RhoReturnMLP, RhoImageDataset
from U_net import FBPConvNet, DeepCormack_dataloader
from torchMCM_functions_new import expand_quadrant_to_full, load_randrhos

# -----------------------------
# Parameters
# -----------------------------
Count_Rate = 200_000_000
rate_M = Count_Rate // 1_000_000
N = 256
max_samples = 2048
batch_size = 16
EPOCHS = 150
learning_rate = 1e-3

# -----------------------------
# Data Loading
# -----------------------------
# Load RDFs and ground truth images (upper-left quadrant only)
base_dir = "/store/LION/gfbl2/DeepCormack_Data"
folder_real = f"{base_dir}/{rate_M}M_Counts/Rho_Measurement"
folder_ideal = f"{base_dir}/TPMD_Ideal"

# Use the same data loading as in DeepCormack_MLP.py

rhos_meas = load_randrhos(
    folder_real,
    xsize=N,
    nproj=5,
    measurement="rho_rand_measurement",
    max_samples=max_samples
)  # (N_samples, 256, 5, 5)

# Load ground truth images (upper-left quadrant only)
ideal_imgs = []
z_vals = []
files = sorted([f for f in os.listdir(folder_ideal) if re.match(r"tpmd_ideal_\\d+\\.txt", f)])
files = files[:max_samples]
for f in files:
    match = re.search(r"(\\d+)", f)
    if not match:
        continue
    idx = int(match.group(1))
    arr = np.loadtxt(os.path.join(folder_ideal, f)).reshape(256, 5, 256).transpose(1, 0, 2)  # (5,256,256)
    ideal_imgs.append(arr)
    z_vals.append(idx)
if not ideal_imgs:
    raise RuntimeError(f"No valid ground truth images found in {folder_ideal}.")
ideal_imgs = torch.tensor(np.stack(ideal_imgs), dtype=torch.float32)  # (N,5,256,256)
z_vals = torch.tensor(z_vals, dtype=torch.float32)
z_vals = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
dataset = RhoImageDataset(rhos_meas.float(), ideal_imgs, z_vals)

# Split dataset
n = len(dataset)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
n_test = n - n_train - n_val
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Save test set filenames to output directory
# Get indices of test set samples
if hasattr(test_set, 'indices'):
    test_indices = test_set.indices
else:
    test_indices = list(range(n_train + n_val, n))
# Save the filenames (from the original file order)
test_files_path = os.path.join(
    f'/store/LION/gfbl2/DeepCormack_Models/NoPretrain/{rate_M}M_({datetime.now().strftime("%Y%m%d_%H%M%S")})',
    f'test_files_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
# But we want to use the same timestamp as output_dir, so set after output_dir is defined

# -----------------------------
# Models
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp = RhoReturnMLP(length=N, channels=5).float().to(device)
# Optionally load pretrained MLP weights here if desired
unet_params = FBPConvNet.default_parameters()
unet = FBPConvNet(model_parameters=unet_params).to(device)

optimizer = torch.optim.Adam(list(mlp.parameters()) + list(unet.parameters()), lr=learning_rate)
criterion = torch.nn.MSELoss()
writer = SummaryWriter(log_dir=f"runs/no_pretrain_unet_{rate_M}")

# -----------------------------
# Helper: Preprocess for UNet
# -----------------------------
def preprocess_for_unet(mlp_imgs, z_vals):
    """
    Args:
        mlp_imgs: (B, 5, 256, 256) - output from MLP for each channel
        z_vals: (B,) - normalized slice indices
    Returns:
        input_unet: (B*5, 2, 256, 256) - for UNet
    """
    B = mlp_imgs.shape[0]
    imgs = mlp_imgs
    # Add slice index as channel, flatten batch and channel dims
    input_list = []
    for i in range(B):
        for s in range(5):
            img = imgs[i, s]  # (256,256)
            z = z_vals[i]
            img = img.unsqueeze(0)  # (1,256,256)
            z_channel = torch.full_like(img, z)
            input_2ch = torch.cat([img, z_channel], dim=0)  # (2,256,256)
            input_list.append(input_2ch)
    input_unet = torch.stack(input_list, dim=0)  # (B*5,2,256,256)
    return input_unet

def preprocess_target(target_imgs):
    """
    Args:
        target_imgs: (B, 5, 256, 256)
    Returns:
        target: (B*5, 1, 256, 256)
    """
    B = target_imgs.shape[0]
    target_list = []
    for i in range(B):
        for s in range(5):
            img = target_imgs[i, s].unsqueeze(0).unsqueeze(0)  # (1,1,256,256)
            target_list.append(img)
    target = torch.cat(target_list, dim=0)  # (B*5,1,256,256)
    return target

# -----------------------------
# Training Loop
# -----------------------------
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'/store/LION/gfbl2/DeepCormack_Models/NoPretrain/{rate_M}M_({timestamp})'
os.makedirs(output_dir, exist_ok=True)

# Save test set filenames to output directory (after output_dir and timestamp are defined)
if hasattr(test_set, 'indices'):
    test_indices = test_set.indices
else:
    test_indices = list(range(n_train + n_val, n))
# Save the actual filenames (from the original dataset order)
test_files_path = os.path.join(output_dir, f'test_files_{timestamp}.txt')
with open(test_files_path, 'w') as f:
    for idx in test_indices:
        # Save the filename of the corresponding measurement (RDF) file
        f.write(f'{files[idx]}\n')

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    mlp.train()
    unet.train()
    train_loss = 0.0
    for meas, ideal, norm, z in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        meas = meas.to(device)  # (B,256,5,5)
        ideal = ideal.to(device)  # (B,5,256,256)
        z = z.to(device)
        norm = norm.to(device)
        optimizer.zero_grad()
        # MLP: reconstruct each channel
        pred_imgs = []
        for c in range(5):
            # For each channel, input (B,256,5) → output (B,256,5)
            pred = mlp(meas[:,:,c,:])  # (B,256,5)
            # Reconstruct to image (B,256,256) using your existing method
            # Here, assume expand_quadrant_to_full is used after UNet, not before
            # For now, treat pred as (B,256,5) → (B,256,256) per channel
            # If you have a function for this, use it here
            # For now, just fill zeros for missing quadrants (upper-left only)
            img = torch.zeros((pred.shape[0], 256, 256), device=pred.device)
            img[:,:,:pred.shape[2]] = pred.permute(0,2,1)  # (B,5,256) → (B,256,5)
            pred_imgs.append(img)
        pred_imgs = torch.stack(pred_imgs, dim=1)  # (B,5,256,256)
        # Preprocess for UNet
        input_unet = preprocess_for_unet(pred_imgs, z)
        target = preprocess_target(ideal)
        # Forward UNet
        output = unet(input_unet)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * meas.size(0)
    train_loss /= len(train_loader.dataset)
    # Validation
    mlp.eval()
    unet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for meas, ideal, norm, z in val_loader:
            meas = meas.to(device)
            ideal = ideal.to(device)
            z = z.to(device)
            pred_imgs = []
            for c in range(5):
                pred = mlp(meas[:,:,c,:])
                img = torch.zeros((pred.shape[0], 256, 256), device=pred.device)
                img[:,:,:pred.shape[2]] = pred.permute(0,2,1)
                pred_imgs.append(img)
            pred_imgs = torch.stack(pred_imgs, dim=1)
            input_unet = preprocess_for_unet(pred_imgs, z)
            target = preprocess_target(ideal)
            output = unet(input_unet)
            loss = criterion(output, target)
            val_loss += loss.item() * meas.size(0)
    val_loss /= len(val_loader.dataset)
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'mlp_state_dict': mlp.state_dict(),
            'unet_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }, os.path.join(output_dir, f'best_model.pth'))

# Save final model
torch.save({
    'mlp_state_dict': mlp.state_dict(),
    'unet_state_dict': unet.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS,
    'val_loss': val_loss
}, os.path.join(output_dir, f'final_model.pth'))
writer.close()

print("Training complete. Best validation loss:", best_val_loss)
