import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1dPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)
        out = (attn * x).sum(dim=1)
        return out

class Attention1dPoolingProjection(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.25) -> None:
        super(Attention1dPoolingProjection, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.final(x)
        return x

class Attention1dPoolingHead(nn.Module):
    """Outputs of the model with the attention1d"""

    def __init__(
        self, hidden_size: int, num_labels: int, dropout: float = 0.25
    ):  # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(Attention1dPoolingHead, self).__init__()
        self.attention1d = Attention1dPooling(hidden_size)
        self.attention1d_projection = Attention1dPoolingProjection(hidden_size, num_labels, dropout)

    def forward(self, x, input_mask=None):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.attention1d_projection(x)
        return x

class MeanPooling(nn.Module):
    """Mean Pooling for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if input_mask is not None:
            # Applying input_mask to zero out masked values
            masked_features = features * input_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / input_mask.sum(dim=1, keepdim=True)
        else:
            mean_pooled_features = torch.mean(features, dim=1)
        return mean_pooled_features


class MeanPoolingProjection(nn.Module):
    """Mean Pooling with a projection layer for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, dropout=0.25):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, mean_pooled_features):
        x = self.dropout(mean_pooled_features)
        x = self.dense(x)
        x = ACT2FN['gelu'](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MeanPoolingHead(nn.Module):
    """Mean Pooling Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, dropout=0.25):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.mean_pooling_projection = MeanPoolingProjection(hidden_size, num_labels, dropout)

    def forward(self, features, input_mask=None):
        mean_pooling_features = self.mean_pooling(features, input_mask=input_mask)
        x = self.mean_pooling_projection(mean_pooling_features)
        return x


class LightAttentionPoolingHead(nn.Module):
    def __init__(self, hidden_size=1280, num_labels=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttentionPoolingHead, self).__init__()

        self.feature_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, num_labels)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, hidden_size] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,num_labels] tensor with logits
        """
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, sequence_length]
        o = self.feature_convolution(x)  # [batch_size, hidden_size, sequence_length]
        o = self.dropout(o)  # [batch_gsize, hidden_size, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, hidden_size, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, hidden_size]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, hidden_size]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*hidden_size]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, num_labels]


# ============= Multi-task regression related classes =============

class MultiTaskRegressionProjection(nn.Module):
    """Multi-task regression projection layer"""
    def __init__(self, hidden_size, num_tasks, dropout=0.25, shared_layers=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.shared_layers = shared_layers
        
        if shared_layers:
            # Shared hidden layers, each task has independent output layer
            self.shared_dense = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.task_heads = nn.ModuleList([
                nn.Linear(hidden_size, 1) for _ in range(num_tasks)
            ])
        else:
            # Each task has independent network
            self.task_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                ) for _ in range(num_tasks)
            ])

    def forward(self, x):
        if self.shared_layers:
            # Shared feature extraction
            x = self.shared_dense(x)
            x = ACT2FN['gelu'](x)  # or use nn.ReLU()
            x = self.dropout(x)
            
            # Independent prediction for each task
            outputs = torch.cat([task_head(x) for task_head in self.task_heads], dim=-1)
        else:
            # Independent processing for each task
            outputs = torch.cat([task_net(x) for task_net in self.task_networks], dim=-1)
        
        return outputs  # [batch_size, num_tasks]


class Attention1dPoolingRegressionProjection(nn.Module):
    def __init__(self, hidden_size, num_tasks, dropout=0.25, shared_layers=True) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.shared_layers = shared_layers
        
        if shared_layers:
            self.linear = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.task_heads = nn.ModuleList([
                nn.Linear(hidden_size, 1) for _ in range(num_tasks)
            ])
        else:
            self.task_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                ) for _ in range(num_tasks)
            ])

    def forward(self, x):
        if self.shared_layers:
            x = self.linear(x)
            x = self.dropout(x)
            x = self.relu(x)
            outputs = torch.cat([task_head(x) for task_head in self.task_heads], dim=-1)
        else:
            outputs = torch.cat([task_net(x) for task_net in self.task_networks], dim=-1)
        return outputs


class Attention1dPoolingRegressionHead(nn.Module):
    """Multi-task regression version of Attention1d pooling head"""
    def __init__(self, hidden_size: int, num_tasks: int, dropout: float = 0.25, shared_layers: bool = True):
        super().__init__()
        self.attention1d = Attention1dPooling(hidden_size)
        self.regression_projection = Attention1dPoolingRegressionProjection(hidden_size, num_tasks, dropout, shared_layers)

    def forward(self, x, input_mask=None):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.regression_projection(x)
        return x


class MeanPoolingRegressionHead(nn.Module):
    """Multi-task regression version of Mean pooling head"""
    def __init__(self, hidden_size, num_tasks, dropout=0.25, shared_layers=True):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.regression_projection = MultiTaskRegressionProjection(hidden_size, num_tasks, dropout, shared_layers)

    def forward(self, features, input_mask=None):
        mean_pooled_features = self.mean_pooling(features, input_mask=input_mask)
        x = self.regression_projection(mean_pooled_features)
        return x


class LightAttentionPoolingRegressionHead(nn.Module):
    """Multi-task regression version of lightweight attention pooling head"""
    def __init__(self, hidden_size=1280, num_tasks=1, dropout=0.25, kernel_size=9, 
                 conv_dropout: float = 0.25, shared_layers=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.shared_layers = shared_layers

        self.feature_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

        # Feature extraction layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        # Multi-task output layer
        if shared_layers:
            # Shared features, independent output heads
            self.task_heads = nn.ModuleList([
                nn.Linear(32, 1) for _ in range(num_tasks)
            ])
        else:
            # Independent output network for each task
            self.task_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                ) for _ in range(num_tasks)
            ])

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, hidden_size]
            mask: [batch_size, sequence_length]
        Returns:
            regression_outputs: [batch_size, num_tasks]
        """
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, sequence_length]
        o = self.feature_convolution(x)
        o = self.dropout(o)
        attention = self.attention_convolution(x)

        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batch_size, hidden_size]
        o2, _ = torch.max(o, dim=-1)  # [batch_size, hidden_size]
        o = torch.cat([o1, o2], dim=-1)  # [batch_size, 2*hidden_size]
        
        # Shared feature extraction
        features = self.feature_extractor(o)  # [batch_size, 32]
        
        # Multi-task output
        if self.shared_layers:
            outputs = torch.cat([task_head(features) for task_head in self.task_heads], dim=-1)
        else:
            outputs = torch.cat([task_net(features) for task_net in self.task_networks], dim=-1)
        
        return outputs  # [batch_size, num_tasks]