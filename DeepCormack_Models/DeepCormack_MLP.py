from torchMCM_functions_new import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import re
import numpy as np

# ============================================================
# Dataset
# ============================================================

class RhoImageDataset(Dataset):
    """
    Each sample:
        meas_rhos   : (256, 5, 5)  # (radius, projection, channel)
        ideal_imgs  : (5, 256, 256) # (channel, radius, radius) upper-left quadrant only
        z           : scalar (normalized slice index)
    """
    def __init__(self, meas_rhos, ideal_imgs, z_vals):
        assert meas_rhos.shape[0] == ideal_imgs.shape[0]
        self.meas = meas_rhos.clone().detach()      # (N,256,5,5)
        self.ideal = ideal_imgs.clone().detach()    # (N,5,256,256)
        self.z = z_vals.clone().detach()            # (N,)

    def __len__(self):
        return self.meas.shape[0]

    def __getitem__(self, idx):
        meas = self.meas[idx]      # (256,5,5)
        ideal = self.ideal[idx]    # (5,256,256)
        z = self.z[idx]
        norm = meas.abs().max()
        norm = norm if norm > 0 else 1.0
        meas = meas / norm
        return meas, ideal, norm, z

# ============================================================
# MLP
# ============================================================

class RhoReturnMLP(nn.Module):
    """
    Fully-connected MLP operating on multi-channel RDFs.
    Input / output: (batch, 256, 5)
    """
    def __init__(self, length, channels=5, hidden_dim=11*256, num_layers=3):
        super().__init__()
        input_dim = length * channels
        layers = [nn.Flatten()]
        last_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        layers += [
            nn.Linear(hidden_dim, input_dim),
            nn.Unflatten(1, (length, channels))
        ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ============================================================
# Training
# ============================================================

count_list = [200_000_000]

for count_ttl in count_list:
    # --------------------------
    # Parameters
    # --------------------------
    N = 513
    xsize = N // 2
    rate_m = count_ttl // 1_000_000
    max_samples = 1500
    batch_size = 256
    learning_rate = 1e-3
    epochs = 3
    base_dir = "/store/LION/gfbl2/DeepCormack_Data"
    folder_real = f"{base_dir}/{rate_m}M_Counts/Rho_Measurement"
    folder_ideal = f"{base_dir}/TPMD_Ideal"

    # --------------------------
    # Load data
    # --------------------------
    rhos_meas = load_randrhos(
        folder_real,
        xsize=xsize,
        nproj=5,
        measurement="rho_rand_measurement",
        max_samples=max_samples
    )  # (N,256,5,5)

    # Ideal images: stored as upper-left quadrant only
    ideal_imgs = []
    z_vals = []
    # Fix regex: use single backslash in raw string
    files = sorted([f for f in os.listdir(folder_ideal) if re.match(r"tpmd_ideal_\d+\.txt", f)])
    if not files:
        print(f"[DEBUG] No files matched in {folder_ideal}.")
        print(f"[DEBUG] Directory contents: {os.listdir(folder_ideal)}")
    files = files[:max_samples]
    for f in files:
        match = re.search(r"(\d+)", f)
        if not match:
            print(f"[WARNING] Could not extract index from filename: {f}")
            continue
        idx = int(match.group(1))
        arr = np.loadtxt(os.path.join(folder_ideal, f)).reshape(256, 5, 256).transpose(1, 0, 2)  # (5,256,256)
        ideal_imgs.append(arr)
        z_vals.append(idx)
    if not ideal_imgs:
        raise RuntimeError(f"No valid ground truth images found in {folder_ideal}. Check file naming and directory structure.")
    # Use float32 for all tensors and model
    ideal_imgs = torch.tensor(np.stack(ideal_imgs), dtype=torch.float32)  # (N,5,256,256)
    z_vals = torch.tensor(z_vals, dtype=torch.float32)
    z_vals = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
    dataset = RhoImageDataset(rhos_meas.float(), ideal_imgs, z_vals)

    # --------------------------
    # Split
    # --------------------------
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # --------------------------
    # Model
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RhoReturnMLP(length=xsize, channels=5).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(log_dir=f"runs/rho_image_mlp_{rate_m}")
    best_val_loss = float("inf")
    model_dir = f"/store/LION/gfbl2/DeepCormack_Models/MLP/{rate_m}M_Counts"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f"mlp_image_best_{rate_m}.pth")
    train_losses = []
    val_losses = []
    n_save_points = 4
    save_epochs = [int(epochs * frac) for frac in [0.25, 0.5, 0.75, 1.0]]
    min_val_loss_after_half = float("inf")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        for meas, ideal_quad, norm, z in train_loader:
            meas = meas.to(device)         # (B,256,5,5)
            ideal_quad = ideal_quad.to(device)  # (B,5,256,256)
            norm = norm.to(device)
            optimizer.zero_grad()
            # For each channel, pass through model and reconstruct
            pred_imgs = []
            for c in range(5):
                pred_rdf = model(meas[..., c]) * norm[:, None, None]
                pred_img = reconstruct_image_batch(pred_rdf.unsqueeze(-1))  # (B,1,512,512)
                pred_imgs.append(pred_img)
            pred_imgs = torch.cat(pred_imgs, dim=1)  # (B,5,512,512)
            ideal_full = expand_quadrant_to_full(ideal_quad)
            pred_imgs = pred_imgs / (pred_imgs.abs().amax(dim=(2,3), keepdim=True) + 1e-8)
            ideal_full = ideal_full / (ideal_full.abs().amax(dim=(2,3), keepdim=True) + 1e-8)
            loss = loss_fn(pred_imgs, ideal_full)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * meas.size(0)
        train_loss /= n_train
        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for meas, ideal_quad, norm, z in val_loader:
                meas = meas.to(device)
                ideal_quad = ideal_quad.to(device)
                norm = norm.to(device)
                pred_imgs = []
                for c in range(5):
                    pred_rdf = model(meas[..., c]) * norm[:, None, None]
                    pred_img = reconstruct_image_batch(pred_rdf.unsqueeze(-1))
                    pred_imgs.append(pred_img)
                pred_imgs = torch.cat(pred_imgs, dim=1)
                ideal_full = expand_quadrant_to_full(ideal_quad)
                pred_imgs = pred_imgs / (pred_imgs.abs().amax(dim=(2,3), keepdim=True) + 1e-8)
                ideal_full = ideal_full / (ideal_full.abs().amax(dim=(2,3), keepdim=True) + 1e-8)
                loss = loss_fn(pred_imgs, ideal_full)
                val_loss += loss.item() * meas.size(0)
        val_loss /= n_val
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Save model at 25%, 50%, 75%, 100% of training
        if (epoch + 1) in save_epochs:
            save_path = os.path.join(model_dir, f"mlp_image_{rate_m}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
        # After 50% of training, save best model so far if val_loss improves
        if epoch + 1 >= epochs // 2:
            if val_loss < min_val_loss_after_half:
                min_val_loss_after_half = val_loss
                torch.save(model.state_dict(), os.path.join(model_dir, f"mlp_image_best_after50pct_{rate_m}.pth"))
        # Save output_mlp.log to model_dir after every epoch
        log_src = os.path.join(os.path.dirname(__file__), "output_mlp.log")
        log_dst = os.path.join(model_dir, "output_mlp.log")
        if os.path.exists(log_src):
            import shutil
            shutil.copy2(log_src, log_dst)
        # Save training and validation loss arrays to model_dir after every epoch
        np.save(os.path.join(model_dir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(model_dir, "val_losses.npy"), np.array(val_losses))
        # Save matplotlib plot of training and validation loss after every epoch
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(model_dir, f'loss_plot_{rate_m}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    torch.save(model.state_dict(), os.path.join(model_dir, f"mlp_image_final_{rate_m}.pth"))
    writer.close()
