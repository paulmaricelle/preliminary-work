import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
import copy

def train_and_evaluate(model, train_loader, X_test, Y_test, optimizer, loss, epochs=50):
    """ Trains a PyTorch model and evaluates it simultaneously """
    
    for epoch in range(epochs):
        
        model.train() 
        epoch_loss = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            batch_loss = loss(y_pred, y)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            
        train_loss = epoch_loss / len(train_loader)
        
        
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            val_loss = loss(test_pred, Y_test).item()
            
        
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch : {epoch+1}/{epochs} , Train Loss : {train_loss} , Val Loss : {val_loss}")



def train_and_evaluate_with_dispersion_plots(model, train_loader, X_test, Y_test, optimizer, loss, epochs):
    """ Trains a model with PyTorch and plots charts on the validation set to get an idea of the variability
    of the model predictions""" 
    for epoch in range(epochs):
        
        model.train() 
        epoch_loss = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            batch_loss = loss(y_pred, y)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            
        train_loss = epoch_loss / len(train_loader)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                val_loss = loss(test_pred, Y_test).item()
            print(f"Epoch : {epoch+1}/{epochs} , Train Loss : {train_loss} , Val Loss : {val_loss}")

    truth_numpy = Y_test.cpu().numpy()
    correlations = []
    test_pred_numpy = test_pred.cpu().numpy()

    n_constant = 0
    for gene in range(truth_numpy.shape[1]):
        if truth_numpy[:, gene].std() < 1e-8 or test_pred_numpy[:, gene].std() < 1e-8:
            correlations.append(0.0)
            n_constant += 1
        else:
            r, _ = pearsonr(test_pred_numpy[:, gene], truth_numpy[:, gene])
            correlations.append(r)

    print(f"Constant genes skipped (r set to 0) : {n_constant}/{truth_numpy.shape[1]}")
        
    correlations = np.array(correlations)
    median_corr = np.median(correlations)
    print(f" Median correlation is {median_corr:.3f}")

    pred_std  = test_pred_numpy.std(axis=0)
    truth_std = truth_numpy.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(truth_std.max(), pred_std.max()) * 1.05
    ax.scatter(truth_std, pred_std, alpha=0.3, s=8, color='steelblue')
    ax.plot([0, lim], [0, lim],         color='black', linestyle='--', linewidth=1,   label='y = x (perfect)')
    ratio_median = np.median(pred_std / (truth_std + 1e-8))
    ax.text(0.05, 0.92, f'Median ratio pred_std / truth_std : {ratio_median:.2f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Real std per gene')
    ax.set_ylabel('Predicted std per gene')
    ax.set_title('Variance compression : predicted vs real')
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correlations, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(0,           color='red',    linestyle='--', linewidth=1.2, label='r = 0 (chance)')
    ax.axvline(median_corr, color='orange', linestyle='--', linewidth=1.2, label=f'Median r = {median_corr:.3f}')
    ax.axvline(0.2,         color='green',  linestyle=':',  linewidth=1,   label=f'r > 0.2 : {(correlations > 0.2).mean()*100:.0f}%')
    ax.axvline(0.4,         color='purple', linestyle=':',  linewidth=1,   label=f'r > 0.4 : {(correlations > 0.4).mean()*100:.0f}%')
    ax.set_xlabel('Correlation (predicted vs real)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Per-gene correlation distribution')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    n_genes = truth_numpy.shape[1]
    idx_best   = int(np.argmax(correlations))
    idx_median = int(np.argsort(correlations)[n_genes // 2])
    idx_worst  = int(np.argmin(correlations))

    panels = [
        (idx_best,   f'Best gene (r = {correlations[idx_best]:.2f})'),
        (idx_median, f'Median gene (r = {correlations[idx_median]:.2f})'),
        (idx_worst,  f'Worst gene (r = {correlations[idx_worst]:.2f})'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (g, title) in zip(axes, panels):
        bins = np.linspace(
            min(truth_numpy[:, g].min(), test_pred_numpy[:, g].min()),
            max(truth_numpy[:, g].max(), test_pred_numpy[:, g].max()),
            25
        )
        axes_flat = ax
        axes_flat.hist(truth_numpy[:, g],     bins=bins, alpha=0.6, color='steelblue', label='Ground truth')
        axes_flat.hist(test_pred_numpy[:, g], bins=bins, alpha=0.6, color='salmon',    label='Predicted')
        axes_flat.set_title(title, fontsize=11)
        axes_flat.set_xlabel('Expression (scaled)')
        axes_flat.set_ylabel('Patients')
        axes_flat.legend(fontsize=8)

    fig.suptitle('Predicted vs real distributions — best / median / worst gene', fontsize=13)
    plt.tight_layout()
    plt.show()


# New functions for more control over the experiments    

def train(model, X_train, y_train, X_test, y_test, optimizer, loss_fn, config):
    """
    Trains a PyTorch model with early stopping and best model checkpointing.
    Stops when validation loss has not improved for `patience` epochs.
    Returns the model loaded with the best weights.
    """

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(config["max_epochs"]):

        # Training
        model.train()
        epoch_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            batch_loss = loss_fn(y_pred, y)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        train_loss = epoch_loss / len(dataloader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_test), y_test).item()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:>4} | Train loss : {train_loss:.4f} | Val loss : {val_loss:.4f} | No improve : {epochs_no_improve}/{config['patience']}")

        # Checkpointing
        if val_loss < best_val_loss - config["min_delta"]:
            best_val_loss     = val_loss
            best_weights      = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config["patience"]:
            print(f"Early stopping at epoch {epoch+1} — best val loss : {best_val_loss:.4f}")
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    return model

def build_fold_data(proteomics_raw, Y, metadata_numeric, train_idx, test_idx, config):
    """
    Prepares one fold of data : imputation, scaling, tensor conversion.
    All preprocessing is fit on train and applied to test to avoid data leakage.
    
    Returns X_train, X_test, Y_train, Y_test as float32 tensors on device.
    """

    # Split
    prot_train, prot_test = proteomics_raw[train_idx], proteomics_raw[test_idx]
    meta_train, meta_test = metadata_numeric[train_idx], metadata_numeric[test_idx]
    y_train, y_test       = Y[train_idx], Y[test_idx]

    # Impute (fit on train only)
    imputer = KNNImputer(n_neighbors=config["n_neighbors"])
    prot_train = imputer.fit_transform(prot_train)
    prot_test  = imputer.transform(prot_test)

    # Combine proteomics + metadata
    X_train = np.hstack([prot_train, meta_train])
    X_test  = np.hstack([prot_test,  meta_test])

    # Scale X (fit on train only)
    scaler_x = RobustScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test  = scaler_x.transform(X_test)

    # Scale Y (fit on train only)
    scaler_y = RobustScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test  = scaler_y.transform(y_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(config["device"])
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(config["device"])
    y_train = torch.tensor(y_train, dtype=torch.float32).to(config["device"])
    y_test  = torch.tensor(y_test,  dtype=torch.float32).to(config["device"])

    return X_train, X_test, y_train, y_test

def run_experiment(model_class, loss_fn, X, Y, groups, config):
    """
    Runs full GroupKFold cross-validation for a given model and loss.
    Returns median_r and ratio_std per fold.
    """

    results = {"median_r": [], "ratio_std": []}
    gkf = GroupKFold(n_splits=config["n_splits"])

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=groups)):
        print(f"\n--- Fold {fold + 1}/{config['n_splits']} ---")

        X_train, X_test, y_train, y_test = build_fold_data(
            X, Y, groups, train_idx, test_idx, config
        )

        model     = model_class(X_train.shape[1], y_train.shape[1]).to(config["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        model = train(model, X_train, y_train, X_test, y_test, optimizer, loss_fn, config)

        results["median_r"].append(compute_median_correlation(model, X_test, y_test))
        results["ratio_std"].append(compute_ratio_std(model, X_test, y_test))

    print(f"\n>>> {np.mean(results['median_r']):.3f} ± {np.std(results['median_r']):.3f} median r")
    return results

def compute_median_correlation(model, X_test, Y_test):
    """Returns median per-gene correlation on the test set. 
    This gives us a loss-independent evaluation metric."""
    from scipy.stats import pearsonr

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
    truth = Y_test.cpu().numpy()

    correlations = []
    for g in range(truth.shape[1]):
        if truth[:, g].std() < 1e-8 or preds[:, g].std() < 1e-8:
            correlations.append(0.0)
        else:
            r, _ = pearsonr(preds[:, g], truth[:, g])
            correlations.append(r)

    return np.median(correlations)

def compute_ratio_std(model, X_test, Y_test):
    """
    Computes the median ratio of predicted std to real std per gene.
    A ratio < 1 indicates variance compression (model predicts toward the mean).
    """

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
    truth = Y_test.cpu().numpy()

    pred_std  = preds.std(axis=0)
    truth_std = truth.std(axis=0)

    non_constant = truth_std > 1e-8
    ratio = pred_std[non_constant] / truth_std[non_constant]

    return float(np.median(ratio))