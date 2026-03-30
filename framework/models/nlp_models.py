import torch
import torch.nn as nn
import torch.nn.functional as F


class _TextCNNEmbed(nn.Module):
    """Embedding + parallel Conv1d + MaxPool feature extractor."""

    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in (3, 4, 5)]
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len]  (int tensor, padded)
        e = self.embedding(x).permute(0, 2, 1)  # [B, embed_dim, seq_len]
        pooled = [F.relu(conv(e)).max(dim=-1).values for conv in self.convs]
        return self.dropout(torch.cat(pooled, dim=1))  # [B, num_filters*3]


class TextCNNAGNews(nn.Module):
    """
    TextCNN for AG News 4-class classification (Kim 2014).
    Embedding(25000, 128) → Conv1d k∈{3,4,5} (100 filters each)
    → MaxPool → concat(300-dim) → Linear(4).
    """

    def __init__(
        self,
        vocab_size: int = 25000,
        embed_dim: int = 128,
        num_filters: int = 100,
        num_classes: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in (3, 4, 5)]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x).permute(0, 2, 1)
        pooled = [F.relu(conv(e)).max(dim=-1).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class SplitTextCNNAGNews(nn.Module):
    """
    Split TextCNN for HCFL/LCFed/FedPer on AG News.
      - embed : Embedding + Conv1d → 300-dim feature
      - head  : Linear(300 → 4)
    """

    def __init__(
        self,
        vocab_size: int = 25000,
        embed_dim: int = 128,
        num_filters: int = 100,
        num_classes: int = 4,
    ):
        super().__init__()
        self.embed = _TextCNNEmbed(vocab_size, embed_dim, num_filters)
        self.head = nn.Linear(num_filters * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
