import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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