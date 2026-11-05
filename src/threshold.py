"""
threshold.py: Banking Business Cost Model and Threshold Calibration.
Locates optimal decision threshold t* = 0.1200 on the Validation Set.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import recall_score, precision_score, fbeta_score


def evaluate_banking_cost(y_true, y_probs, amounts, threshold, fp_cost=5.0, fn_multiplier=1.0):
    """
    Computes total operational banking cost:
    Total Cost = Σ(FN_Multiplier * Amount for missed frauds) + (FP_Cost * N_false_alarms).
    """
    preds = (y_probs >= threshold).astype(int)
    fn_mask = (y_true == 1) & (preds == 0)
    fp_mask = (y_true == 0) & (preds == 1)

    cost_fn = float(np.sum(amounts[fn_mask] * fn_multiplier))
    cost_fp = float(np.sum(fp_mask) * fp_cost)
    total_cost = cost_fn + cost_fp

    return {
        'total_cost': total_cost,
        'cost_fn': cost_fn,
        'cost_fp': cost_fp,
        'n_fn': int(np.sum(fn_mask)),
        'n_fp': int(np.sum(fp_mask)),
        'n_tp': int(np.sum((y_true == 1) & (preds == 1))),
        'n_tn': int(np.sum((y_true == 0) & (preds == 0)))
    }


def find_optimal_threshold(y_val, val_probs, val_amounts, thresholds=None, fp_cost=5.0):
    """
    Sweeps thresholds on the validation set to locate the cost-minimizing threshold t*.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.90, 200)

    costs = []
    recalls = []
    precisions = []
    f2s = []

    for t in thresholds:
        res = evaluate_banking_cost(y_val, val_probs, val_amounts, threshold=t, fp_cost=fp_cost)
        costs.append(res['total_cost'])
        preds = (val_probs >= t).astype(int)
        recalls.append(recall_score(y_val, preds, zero_division=0))
        precisions.append(precision_score(y_val, preds, zero_division=0))
        f2s.append(fbeta_score(y_val, preds, beta=2, zero_division=0))

    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    opt_t = float(thresholds[opt_idx])

    # Default t=0.50 comparison
    def_idx = np.abs(thresholds - 0.50).argmin()
    default_cost = float(costs[def_idx])
    min_cost = float(costs[opt_idx])
    dollars_saved = default_cost - min_cost

    summary = {
        'optimal_threshold': opt_t,
        'minimum_cost': min_cost,
        'default_cost': default_cost,
        'dollars_saved': dollars_saved,
        'thresholds': thresholds,
        'costs': costs,
        'recalls': recalls,
        'precisions': precisions,
        'f2s': f2s
    }
    return summary


def plot_cost_curve(threshold_summary, save_path=None):
    """
    Renders and saves dual-panel cost curve and metric tradeoff chart at 150 DPI.
    """
    thresholds = threshold_summary['thresholds']
    costs = threshold_summary['costs']
    recalls = threshold_summary['recalls']
    precisions = threshold_summary['precisions']
    f2s = threshold_summary['f2s']
    opt_t = threshold_summary['optimal_threshold']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)

    # Panel 1: Cost Curve
    ax1.plot(thresholds, costs, color='#d9383a', lw=2.5, label='Total Operational Cost')
    ax1.axvline(x=opt_t, color='black', linestyle='--', label=f'Optimal t* = {opt_t:.4f}')
    ax1.axvline(x=0.50, color='gray', linestyle=':', label='Default t = 0.5000')
    ax1.set_title('Operational Cost vs Decision Threshold (Validation Set)', fontweight='bold')
    ax1.set_xlabel('Decision Threshold (t)')
    ax1.set_ylabel('Total Operational Cost ($)')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${int(x):,}'))
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Metric Tradeoff
    ax2.plot(thresholds, recalls, color='#d9383a', lw=2, label='Recall (Fraud Coverage)')
    ax2.plot(thresholds, precisions, color='#2b5c8f', lw=2, label='Precision')
    ax2.plot(thresholds, f2s, color='darkgreen', lw=2, linestyle='--', label='F2 Score')
    ax2.axvline(x=opt_t, color='black', linestyle='--', label=f'Operating Point (t* = {opt_t:.2f})')
    ax2.set_title('Operational Tradeoff: Recall vs Precision vs F2', fontweight='bold')
    ax2.set_xlabel('Decision Threshold (t)')
    ax2.set_ylabel('Metric Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
