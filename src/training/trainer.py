import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from .scheduler import create_scheduler
from .metrics import setup_metrics
from .loss_function import MultiClassFocalLossWithAlpha
import wandb
from models.model_factory import create_plm_and_tokenizer
from peft import PeftModel

class Trainer:
    def __init__(self, args, model, plm_model, logger, train_loader):
        self.args = args
        self.model = model
        self.plm_model = plm_model
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loader = train_loader
        
        # Setup metrics
        self.metrics_dict = setup_metrics(args)
        
        # Setup optimizer with different learning rates
        if self.args.training_method == 'full':
            # Use a smaller learning rate for PLM
            optimizer_grouped_parameters = [
                {
                    "params": self.model.parameters(),
                    "lr": args.learning_rate
                },
                {
                    "params": self.plm_model.parameters(),
                    "lr": args.learning_rate
                }
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6, weight_decay=0.01)
        elif self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
            optimizer_grouped_parameters = [
                {
                    "params": self.model.parameters(),
                    "lr": args.learning_rate                    
                },
                {
                    "params": [param for param in self.plm_model.parameters() if param.requires_grad],
                    "lr": args.learning_rate
                }
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6, weight_decay=0.01)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, eps=1e-6, weight_decay=0.01)
        
        # Setup accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        
        # Setup scheduler
        self.scheduler = create_scheduler(args, self.optimizer, self.train_loader)
        
        # Setup loss function
        self.loss_fn = self._setup_loss_function()
        
        # Prepare for distributed training
        if self.args.training_method in ['full', 'plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
            self.model, self.plm_model, self.optimizer = self.accelerator.prepare(
                self.model, self.plm_model, self.optimizer
            )
        else:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)
            
        # Training state
        self.best_val_loss = float("inf")
        if self.args.monitor_strategy == 'min':
            self.best_val_metric_score = float("inf")
        else:
            self.best_val_metric_score = -float("inf")
        self.global_steps = 0
        self.early_stop_counter = 0
        
        # Save args
        with open(os.path.join(self.args.output_dir, f'{self.args.output_model_name.split(".")[0]}.json'), 'w') as f:
            json.dump(self.args.__dict__, f)
        
    def _setup_loss_function(self):
        if self.args.problem_type == 'regression':
            return torch.nn.MSELoss()
        elif self.args.problem_type == 'multi_label_classification':
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()
            
    def _compute_loss_multitask_regression(self, logits, labels):
        """Multi-task regression loss computation, ignoring NaN values"""
        # Check input validity
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        valid_mask = ~torch.isnan(labels)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute loss for each task
        task_losses = []
        for task_idx in range(logits.shape[1]):
            task_valid_mask = valid_mask[:, task_idx]
            if task_valid_mask.any():
                task_logits = logits[task_valid_mask, task_idx]
                task_labels = labels[task_valid_mask, task_idx]
                task_loss = F.mse_loss(task_logits, task_labels, reduction='mean')
                
                if torch.isnan(task_loss) or torch.isinf(task_loss):
                    task_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                task_losses.append(task_loss)
        
        if not task_losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Simple averaging
        final_loss = torch.stack(task_losses).mean()
        
        # Loss clipping
        final_loss = torch.clamp(final_loss, max=100.0)
        return final_loss
    
    def train(self, train_loader, val_loader):
        """Train the model."""
        for epoch in range(self.args.num_epochs):
            self.logger.info(f"---------- Epoch {epoch} ----------")
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.logger.info(f'Epoch {epoch} Train Loss: {train_loss:.4f}')
            
            # Validation phase
            val_loss, val_metrics = self._validate(val_loader)
            
            # Handle validation results (model saving, early stopping)
            self._handle_validation_results(epoch, val_loss, val_metrics)
            
            # Early stopping check
            if self._check_early_stopping():
                self.logger.info(f"Early stop at Epoch {epoch}")
                break
                
    def _train_epoch(self, train_loader):
        self.model.train()
        if self.args.training_method in  ['full', 'plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
            self.plm_model.train()
        total_loss = 0
        total_samples = 0
        epoch_iterator = tqdm(train_loader, desc="Training")
        
        for batch in epoch_iterator:
            # choose models to accumulate
            models_to_accumulate = [self.model, self.plm_model] if self.args.training_method in  ['full', 'plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3'] else [self.model]
            
            with self.accelerator.accumulate(*models_to_accumulate):
                # Forward and backward
                loss = self._training_step(batch)
                self.accelerator.backward(loss)
                    
                # Update statistics
                batch_size = batch["label"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Gradient clipping if needed
                if self.args.max_grad_norm > 0:
                    params_to_clip = (
                        list(self.model.parameters()) + list(self.plm_model.parameters())
                        if self.args.training_method in  ['full', 'plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']
                        else self.model.parameters()
                    )
                    self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                
                # Optimization step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                self.global_steps += 1
                self._log_training_step(loss)
                
                # Update progress bar
                epoch_iterator.set_postfix(
                    train_loss=loss.item(),
                    grad_step=self.global_steps // self.args.gradient_accumulation_steps
                )
        
        return total_loss / total_samples
    
    def _training_step(self, batch):
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        logits = self.model(self.plm_model, batch)
        # print('----------------------------------')
        # print(f'logits:{logits.shape}')
        # print(f'label:{batch["label"].shape}')
        # print('----------------------------------')
        loss = self._compute_loss(logits, batch["label"])
        
        return loss
    
    def _validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            tuple: (validation_loss, validation_metrics)
        """
        self.model.eval()
        if self.args.training_method in  ['full', 'plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
            self.plm_model.eval()
            
        total_loss = 0
        total_samples = 0
        
        # Reset all metrics at the start of validation
        for metric in self.metrics_dict.values():
            if hasattr(metric, 'reset'):
                metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(self.plm_model, batch)
                loss = self._compute_loss(logits, batch["label"])
                
                # Update loss statistics
                batch_size = len(batch["label"])
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update metrics
                self._update_metrics(logits, batch["label"])
        
        # Compute average loss
        avg_loss = total_loss / total_samples
        
        # Compute final metrics
        metrics_results = self._compute_final_metrics()
        
        return avg_loss, metrics_results
    
    def test(self, test_loader):
        # Load best model
        self._load_best_model()
        
        # Add a clear signal that testing is starting
        self.logger.info("---------- Starting Test Phase ----------")
        
        # Run evaluation with a custom testing function instead of reusing _validate
        test_loss, test_metrics = self._test_evaluate(test_loader)
        
        # Log results
        self.logger.info("Test Results:")
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        for name, value in test_metrics.items():
            self.logger.info(f"Test {name}: {value:.4f}")
            
        if self.args.wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.log({"test/loss": test_loss})
    
    def _test_evaluate(self, test_loader):
        """
        Dedicated evaluation function for test phase with proper labeling.
        This is almost identical to _validate but with "Testing" progress bar.
        """
        self.model.eval()
        if self.args.training_method in ['full', 'plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
            self.plm_model.eval()
            
        total_loss = 0
        total_samples = 0
        
        # Reset all metrics at the start of testing
        for metric in self.metrics_dict.values():
            if hasattr(metric, 'reset'):
                metric.reset()
        
        with torch.no_grad():
            # Note the desc is "Testing" instead of "Validating"
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(self.plm_model, batch)
                loss = self._compute_loss(logits, batch["label"])
                
                # Update loss statistics
                batch_size = len(batch["label"])
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update metrics
                self._update_metrics(logits, batch["label"])
        
        # Compute average loss
        avg_loss = total_loss / total_samples
        
        # Compute final metrics
        metrics_results = self._compute_final_metrics()
        
        return avg_loss, metrics_results
    
    def _compute_loss(self, logits, labels):
        """Compute loss function with support for NaN value handling in multi-task regression"""
        if self.args.problem_type == 'regression':
            if self.args.num_labels == 1:
                return self.loss_fn(logits.squeeze(), labels.squeeze())
            else:
                # Multi-task regression
                return self._compute_loss_multitask_regression(logits, labels.float())
        elif self.args.problem_type == 'multi_label_classification':
            return self.loss_fn(logits, labels.float())
        else:
            return self.loss_fn(logits, labels)
    
    def _update_metrics(self, logits, labels):
        """Update metrics with current batch predictions."""
        # Check if this is multi-task regression
        if (self.args.problem_type == 'regression' and 
            hasattr(self.args, 'num_labels') and 
            self.args.num_labels > 1 and 
            'multitask_regression' in self.metrics_dict):
            
            # Multi-task regression case
            multitask_metric = self.metrics_dict['multitask_regression']
            multitask_metric.update(logits, labels.float())
            
        else:
            # Original single-task logic
            for metric_name, metric in self.metrics_dict.items():
                if self.args.problem_type == 'regression':
                    if self.args.num_labels == 1:
                        logits = logits.view(-1, 1)
                        labels = labels.view(-1, 1)
                        metric(logits, labels)
                    else:
                        # Multi-task regression, compute metrics only at valid positions
                        valid_mask = ~torch.isnan(labels)
                        if valid_mask.any():
                            metric(logits, labels)
                elif self.args.problem_type == 'multi_label_classification':
                    metric(torch.sigmoid(logits), labels)
                else:
                    if self.args.num_labels == 2:
                        if metric_name == 'auroc':
                            metric(torch.sigmoid(logits[:, 1]), labels)
                        else:
                            metric(torch.argmax(logits, 1), labels)
                    else:
                        if metric_name == 'auroc':
                            metric(F.softmax(logits, dim=1), labels)
                        else:
                            metric(logits, labels)
    
    def _compute_final_metrics(self):
        """Compute final metrics results"""
        metrics_results = {}
        
        # Check if this is multi-task regression
        if (self.args.problem_type == 'regression' and 
            hasattr(self.args, 'num_labels') and 
            self.args.num_labels > 1 and 
            'multitask_regression' in self.metrics_dict):
            
            # Multi-task regression case
            multitask_metric = self.metrics_dict['multitask_regression']
            metrics_results = multitask_metric.compute()
            
        else:
            # Original single-task logic
            metrics_results = {name: metric.compute().item() 
                              for name, metric in self.metrics_dict.items() 
                              if hasattr(metric, 'compute')}
        
        return metrics_results
    
    def _log_training_step(self, loss):
        if self.args.wandb:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=self.global_steps)
    
    def _save_model(self, path):
        if self.args.training_method in ['full', 'lora']:
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            plm_state = {k: v.cpu() for k, v in self.plm_model.state_dict().items()}
            torch.save({
                'model_state_dict': model_state,
                'plm_state_dict': plm_state
            }, path)
        elif self.args.training_method == "plm-lora":
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_lora_path = path.replace('.pt', '_lora')
            self.plm_model.save_pretrained(plm_lora_path)
        elif self.args.training_method == "plm-qlora":
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_qlora_path = path.replace('.pt', '_qlora')
            self.plm_model.save_pretrained(plm_qlora_path)
        elif self.args.training_method == "plm-dora":
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_dora_path = path.replace('.pt', '_dora')
            self.plm_model.save_pretrained(plm_dora_path)
        elif self.args.training_method == "plm-adalora":
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_adalora_path = path.replace('.pt', '_adalora')
            self.plm_model.save_pretrained(plm_adalora_path)
        elif self.args.training_method == "plm-ia3":
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_ia3_path = path.replace('.pt', '_ia3')
            self.plm_model.save_pretrained(plm_ia3_path)
        else:
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)

    def _load_best_model(self):
        path = os.path.join(self.args.output_dir, self.args.output_model_name)
        if self.args.training_method in ['full', 'lora']:
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.plm_model.load_state_dict(checkpoint['plm_state_dict'])
            self.model.to(self.device)
            self.plm_model.to(self.device)
        elif self.args.training_method == "plm-lora":
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_lora_path = path.replace('.pt', '_lora')
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_lora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
        elif self.args.training_method == "plm-qlora":
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_qlora_path = path.replace('.pt', '_qlora')
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_qlora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
        elif self.args.training_method == "plm-dora":
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_dora_path = path.replace('.pt', '_dora')
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_dora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
        elif self.args.training_method == "plm-adalora":
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_adalora_path = path.replace('.pt', '_adalora')
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_adalora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
        elif self.args.training_method == "plm-ia3":
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_ia3_path = path.replace('.pt', '_ia3')
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_ia3_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
        else:
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
    
    def _handle_validation_results(self, epoch: int, val_loss: float, val_metrics: dict):
        """
        Handle validation results, including model saving and early stopping checks.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            val_metrics: Dictionary of validation metrics
        """
        # Log validation results
        self.logger.info(f'Epoch {epoch} Val Loss: {val_loss:.4f}')
        
        # For multi-task regression, log detailed results
        if (self.args.problem_type == 'regression' and 
            hasattr(self.args, 'num_labels') and 
            self.args.num_labels > 1):
            
            # Log individual task metrics
            task_names = ['phopt', 'phmin', 'phmax'] if self.args.num_labels == 3 else [f'task_{i}' for i in range(self.args.num_labels)]
            metric_names = ['spcc', 'pcc', 'r2', 'rmse', 'mae', 'mse']
            
            # Log per-task metrics
            for task_name in task_names:
                self.logger.info(f'--- {task_name.upper()} Metrics ---')
                for metric_name in metric_names:
                    key = f'{task_name}_{metric_name}'
                    if key in val_metrics:
                        self.logger.info(f'Epoch {epoch} Val {key}: {val_metrics[key]:.4f}')
            
            # Log average metrics
            self.logger.info('--- Average Metrics ---')
            for metric_name in metric_names:
                key = f'avg_{metric_name}'
                if key in val_metrics:
                    self.logger.info(f'Epoch {epoch} Val {key}: {val_metrics[key]:.4f}')
        else:
            # Original single-task logging
            for metric_name, metric_value in val_metrics.items():
                self.logger.info(f'Epoch {epoch} Val {metric_name}: {metric_value:.4f}')
        
        if self.args.wandb:
            wandb.log({
                "val/loss": val_loss,
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }, step=self.global_steps)
        
        # Check if we should save the model
        should_save = False
        monitor_value = val_loss
        
        # For multi-task regression, use average MSE as monitor metric if not specified
        if (self.args.problem_type == 'regression' and 
            hasattr(self.args, 'num_labels') and 
            self.args.num_labels > 1):
            
            if self.args.monitor == 'loss':
                monitor_value = val_loss
            elif 'avg_mse' in val_metrics:
                monitor_value = val_metrics['avg_mse']
                self.args.monitor = 'avg_mse'  # Update monitor to avg_mse
                self.args.monitor_strategy = 'min'  # MSE should be minimized
            elif self.args.monitor in val_metrics:
                monitor_value = val_metrics[self.args.monitor]
        else:
            # If monitoring a specific metric for single-task
            if self.args.monitor != 'loss' and self.args.monitor in val_metrics:
                monitor_value = val_metrics[self.args.monitor]
        
        # Check if current result is better
        if self.args.monitor_strategy == 'min':
            if monitor_value < self.best_val_metric_score:
                should_save = True
                self.best_val_metric_score = monitor_value
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        else:  # strategy == 'max'
            if monitor_value > self.best_val_metric_score:
                should_save = True
                self.best_val_metric_score = monitor_value
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        
        # Save model if improved
        if should_save:
            self.logger.info(f"Saving model with best val {self.args.monitor}: {monitor_value:.4f}")
            save_path = os.path.join(self.args.output_dir, self.args.output_model_name)
            self._save_model(save_path)

    def _check_early_stopping(self) -> bool:
        """
        Check if training should be stopped early.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.args.patience > 0 and self.early_stop_counter >= self.args.patience:
            self.logger.info(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement")
            return True
        return False