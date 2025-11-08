"""
Graph Neural Network (GIN) for AGV solution encoding

Implements Graph Isomorphism Network layers for processing solution graphs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLP(nn.Module):
    """Multi-layer perceptron for GIN"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        """
        Initialize MLP

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            activation: Activation function ('relu', 'leaky_relu', 'tanh')
        """
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.activation_name = activation

        # Build layers
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:  # No activation on last layer
                if self.activation_name == 'relu':
                    x = F.relu(x)
                elif self.activation_name == 'leaky_relu':
                    x = F.leaky_relu(x, 0.2)
                elif self.activation_name == 'tanh':
                    x = torch.tanh(x)
        return x


class GINConv(nn.Module):
    """
    Graph Isomorphism Network convolution layer

    Implements: h_v' = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        eps: float = 0.0,
        train_eps: bool = False
    ):
        """
        Initialize GIN convolution

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            eps: Initial epsilon value (controls self-loop importance)
            train_eps: Whether epsilon is trainable
        """
        super(GINConv, self).__init__()

        self.mlp = MLP(input_dim, output_dim * 2, output_dim, num_layers=2)

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer('eps', torch.tensor(eps))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)

        Returns:
            Updated node features (num_nodes, output_dim)
        """
        # Extract source and target nodes
        row, col = edge_index

        # Aggregate neighbor features
        if edge_weight is not None:
            # Weighted aggregation
            messages = x[col] * edge_weight.unsqueeze(1)
        else:
            messages = x[col]

        # Sum aggregation
        out = torch.zeros_like(x[:, :messages.shape[1]] if messages.shape[1] < x.shape[1] else x)
        out = out.to(x.device)

        if messages.shape[0] > 0:
            # Scatter add for aggregation
            out = torch.zeros(x.shape[0], messages.shape[1], device=x.device)
            out.index_add_(0, row, messages)

        # Add self-connection
        out = (1 + self.eps) * x[:, :out.shape[1]] + out if out.shape[1] == x.shape[1] else \
              (1 + self.eps) * x + out if out.shape[1] < x.shape[1] else out

        # Apply MLP
        out = self.mlp(out)

        return out


class GIN(nn.Module):
    """
    Graph Isomorphism Network for AGV solution encoding
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize GIN

        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of GIN layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(GIN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GIN layers
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for i in range(num_layers):
            self.gin_layers.append(
                GINConv(hidden_dim, hidden_dim, eps=0.0, train_eps=True)
            )
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch assignment for nodes (num_nodes,)

        Returns:
            Node embeddings (num_nodes, output_dim) or graph embedding (batch_size, output_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Store intermediate representations for skip connections
        xs = [x]

        # Apply GIN layers
        for i, gin_layer in enumerate(self.gin_layers):
            x = gin_layer(x, edge_index)

            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            xs.append(x)

        # Skip connections: concatenate all layer outputs
        x = torch.cat(xs, dim=1)

        # Final projection
        x = self.output_proj(x)

        # Global pooling if batch is provided
        if batch is not None:
            # Mean pooling over nodes in each graph
            batch_size = batch.max().item() + 1
            out = torch.zeros(batch_size, self.output_dim, device=x.device)

            for i in range(batch_size):
                mask = batch == i
                out[i] = x[mask].mean(dim=0)

            return out
        else:
            return x


class GINWithGlobalFeatures(nn.Module):
    """
    GIN that also incorporates global solution features
    """

    def __init__(
        self,
        node_input_dim: int = 8,
        global_input_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize GIN with global features

        Args:
            node_input_dim: Node feature dimension
            global_input_dim: Global feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of GIN layers
            dropout: Dropout rate
        """
        super(GINWithGlobalFeatures, self).__init__()

        self.gin = GIN(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Global feature processing
        self.global_proj = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Combine node and global features
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with global features

        Args:
            x: Node features
            edge_index: Edge indices
            global_features: Global solution features (batch_size, global_input_dim)
            batch: Batch assignment

        Returns:
            Combined embeddings
        """
        # Process graph structure
        node_embed = self.gin(x, edge_index, batch)

        # Process global features
        global_embed = self.global_proj(global_features)

        # Combine
        combined = torch.cat([node_embed, global_embed], dim=1)
        output = self.combiner(combined)

        return output
