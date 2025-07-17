import torch
import torch.optim as optim
import torch.nn as nn
 
def train_model(train_dl, val_dl, model, epochs=10, lr=1e-3, device='cuda',
                log_file="training_log.txt", best_model_path="best_model.pth", final_model_path="final_model.pth"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None
 
    with open(log_file, "w") as f:
        for epoch in range(1, epochs+1):
            # ---- TRAINING PHASE ----
            model.train()
            total_loss = 0.0
            total_coeff_loss = None  # Accumulate per-coefficient losses.
            total_count = 0
 
            for coords, conds, targets in train_dl:
                coords  = coords.to(device)
                conds   = conds.to(device)
                targets = targets.to(device)
 
                preds = model(coords, conds)  # Forward pass.
                batch_loss = criterion(preds, targets)
                
                # Compute per-coefficient squared error.
                batch_squared_error = (preds - targets) ** 2  
                batch_coeff_loss = batch_squared_error.sum(dim=0)
 
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
 
                batch_size = coords.size(0)
                total_loss += batch_loss.item() * batch_size
                total_count += batch_size
 
                if total_coeff_loss is None:
                    total_coeff_loss = batch_coeff_loss
                else:
                    total_coeff_loss += batch_coeff_loss
 
            train_mse = total_loss / total_count
            train_per_coeff_mse = total_coeff_loss / total_count
 
            # ---- VALIDATION PHASE ----
            model.eval()
            val_loss = 0.0
            val_coeff_loss = None
            val_count = 0
 
            with torch.no_grad():
                for coords, conds, targets in val_dl:
                    coords  = coords.to(device)
                    conds   = conds.to(device)
                    targets = targets.to(device)
 
                    preds = model(coords, conds)
                    batch_loss = criterion(preds, targets)
                    
                    batch_squared_error = (preds - targets) ** 2
                    batch_coeff_loss = batch_squared_error.sum(dim=0)
 
                    batch_size = coords.size(0)
                    val_loss += batch_loss.item() * batch_size
                    val_count += batch_size
 
                    if val_coeff_loss is None:
                        val_coeff_loss = batch_coeff_loss
                    else:
                        val_coeff_loss += batch_coeff_loss
 
            val_mse = val_loss / val_count
            val_per_coeff_mse = val_coeff_loss / val_count
 
            # ---- SAVE BEST MODEL (for logging or further use) ----
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_model_state = model.state_dict()
                torch.save(best_model_state, best_model_path)
 
            # ---- LOGGING AND SAVE FINAL MODEL EVERY 100 EPOCHS (and at epoch 1) ----
            if epoch % 100 == 0 or epoch == 1:
                log_str = (f"Epoch {epoch}/{epochs} | "
                           f"Train MSE: {train_mse:.6f} "
                           f"(cp: {train_per_coeff_mse[0]:.6f}, cf_x: {train_per_coeff_mse[1]:.6f}, "
                           f"cf_z: {train_per_coeff_mse[2]:.6f}) | "
                           f"Val MSE: {val_mse:.6f} "
                           f"(cp: {val_per_coeff_mse[0]:.6f}, cf_x: {val_per_coeff_mse[1]:.6f}, "
                           f"cf_z: {val_per_coeff_mse[2]:.6f})")
                print(log_str)
                f.write(log_str + "\n")
                torch.save(model.state_dict(), final_model_path)
 
        print("Training complete. Best Validation MSE:", best_val_loss)
    
    # Return the final model (last saved state)
    return model
