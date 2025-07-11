import torch
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef, MeanSquaredError, MeanAbsoluteError, R2Score, PearsonCorrCoef
from torchmetrics.classification import MultilabelAveragePrecision
import torch.nn.functional as F


def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold, Copied from TorchDrug.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """

    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


class MultilabelF1Max(MultilabelAveragePrecision):

    def compute(self):
        return count_f1_max(torch.cat(self.preds), torch.cat(self.target))


class RMSEMetric(torch.nn.Module):
    """Root Mean Square Error metric"""
    def __init__(self):
        super().__init__()
        self.mse = MeanSquaredError()
    
    def __call__(self, preds, target):
        return self.mse(preds, target)
    
    def compute(self):
        return torch.sqrt(self.mse.compute())
    
    def reset(self):
        self.mse.reset()
    
    def to(self, device):
        self.mse = self.mse.to(device)
        return self


class MultiTaskRegressionMetrics:
    """Multi-task regression metrics container for handling NaN values and per-task computation"""
    
    def __init__(self, num_tasks, task_names=None, device='cpu'):
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]
        self.device = device
        
        # Initialize metrics for each task
        self.task_metrics = {}
        for task_idx in range(num_tasks):
            task_name = self.task_names[task_idx]
            self.task_metrics[task_name] = {
                'spcc': SpearmanCorrCoef().to(device),
                'pcc': PearsonCorrCoef().to(device),
                'r2': R2Score().to(device),
                'rmse': RMSEMetric().to(device),
                'mae': MeanAbsoluteError().to(device),
                'mse': MeanSquaredError().to(device)
            }
    
    def update(self, preds, targets):
        """Update metrics with predictions and targets, handling NaN values"""
        # preds: [batch_size, num_tasks]
        # targets: [batch_size, num_tasks]
        
        for task_idx in range(self.num_tasks):
            task_name = self.task_names[task_idx]
            
            # Get valid indices (not NaN)
            valid_mask = ~torch.isnan(targets[:, task_idx])
            
            if valid_mask.sum() > 0:  # Only update if we have valid samples
                task_preds = preds[valid_mask, task_idx]
                task_targets = targets[valid_mask, task_idx]
                
                # Update all metrics for this task
                for metric_name, metric in self.task_metrics[task_name].items():
                    try:
                        metric(task_preds, task_targets)
                    except Exception as e:
                        print(f"Error updating {metric_name} for {task_name}: {e}")
    
    def compute(self):
        """Compute all metrics for all tasks and their averages"""
        results = {}
        
        # Compute metrics for each task
        task_results = {}
        for task_name in self.task_names:
            task_results[task_name] = {}
            for metric_name, metric in self.task_metrics[task_name].items():
                try:
                    value = metric.compute()
                    if torch.isnan(value) or torch.isinf(value):
                        value = torch.tensor(0.0, device=self.device)
                    task_results[task_name][metric_name] = value.item()
                    results[f'{task_name}_{metric_name}'] = value.item()
                except Exception as e:
                    print(f"Error computing {metric_name} for {task_name}: {e}")
                    task_results[task_name][metric_name] = 0.0
                    results[f'{task_name}_{metric_name}'] = 0.0
        
        # Compute averages across tasks
        metric_names = ['spcc', 'pcc', 'r2', 'rmse', 'mae', 'mse']
        for metric_name in metric_names:
            values = [task_results[task_name][metric_name] for task_name in self.task_names]
            avg_value = sum(values) / len(values)
            results[f'avg_{metric_name}'] = avg_value
        
        return results
    
    def reset(self):
        """Reset all metrics"""
        for task_name in self.task_names:
            for metric in self.task_metrics[task_name].values():
                metric.reset()
    
    def to(self, device):
        """Move metrics to device"""
        self.device = device
        for task_name in self.task_names:
            for metric_name, metric in self.task_metrics[task_name].items():
                self.task_metrics[task_name][metric_name] = metric.to(device)
        return self


def setup_metrics(args):
    """Setup metrics based on problem type and specified metrics list."""
    metrics_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # For multi-task regression, use the special multi-task metrics
    if args.problem_type == 'regression' and hasattr(args, 'num_labels') and args.num_labels > 1:
        # Define task names
        task_names = ['phmin', 'phopt', 'phmax'] if args.num_labels == 3 else [f'task_{i}' for i in range(args.num_labels)]
        
        # Create multi-task regression metrics
        multitask_metrics = MultiTaskRegressionMetrics(
            num_tasks=args.num_labels,
            task_names=task_names,
            device=device
        )
        metrics_dict['multitask_regression'] = multitask_metrics
        
    else:
        # Original single-task logic
        for metric_name in args.metrics:
            if args.problem_type == 'regression':
                metric_config = _setup_regression_metrics(metric_name, device)
            elif args.problem_type == 'single_label_classification':
                if args.num_labels == 2:
                    metric_config = _setup_binary_metrics(metric_name, device)
                else:
                    metric_config = _setup_multiclass_metrics(metric_name, args.num_labels, device)            
            elif args.problem_type == 'multi_label_classification':
                metric_config = _setup_multilabel_metrics(metric_name, args.num_labels, device)
                
            if metric_config:
                metrics_dict[metric_name] = metric_config['metric']
    
    # Add loss to metrics if it's the monitor metric
    if args.monitor == 'loss':
        metrics_dict['loss'] = 'loss'
        
    return metrics_dict

def _setup_regression_metrics(metric_name, device):
    metrics_config = {
        'spearman_corr': {
            'metric': SpearmanCorrCoef().to(device),
        },
        'spcc': {
            'metric': SpearmanCorrCoef().to(device),
        },
        'pcc': {
            'metric': PearsonCorrCoef().to(device),
        },
        'r2': {
            'metric': R2Score().to(device),
        },
        'rmse': {
            'metric': RMSEMetric().to(device),
        },
        'mae': {
            'metric': MeanAbsoluteError().to(device),
        },
        'mse': {
            'metric': MeanSquaredError().to(device),
        }
    }
    return metrics_config.get(metric_name)

def _setup_multiclass_metrics(metric_name, num_classes, device):
    metrics_config = {
        'accuracy': {
            'metric': Accuracy(task='multiclass', num_classes=num_classes).to(device),
        },
        'recall': {
            'metric': Recall(task='multiclass', num_classes=num_classes).to(device),
        },
        'precision': {
            'metric': Precision(task='multiclass', num_classes=num_classes).to(device),
        },
        'f1': {
            'metric': F1Score(task='multiclass', num_classes=num_classes).to(device),
        },
        'mcc': {
            'metric': MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device),
        },
        'auroc': {
            'metric': AUROC(task='multiclass', num_classes=num_classes).to(device),
        }
    }
    return metrics_config.get(metric_name)

def _setup_binary_metrics(metric_name, device):
    metrics_config = {
        'accuracy': {
            'metric': BinaryAccuracy().to(device),
        },
        'recall': {
            'metric': BinaryRecall().to(device),
        },
        'precision': {
            'metric': BinaryPrecision().to(device),
        },
        'f1': {
            'metric': BinaryF1Score().to(device),
        },
        'mcc': {
            'metric': BinaryMatthewsCorrCoef().to(device),
        },
        'auroc': {
            'metric': BinaryAUROC().to(device),
        }
    }
    return metrics_config.get(metric_name)

def _setup_multilabel_metrics(metric_name, num_labels, device):
    metrics_config = {
        'f1_max': {
            'metric': MultilabelF1Max(num_labels=num_labels).to(device),
        }
    }
    return metrics_config.get(metric_name)