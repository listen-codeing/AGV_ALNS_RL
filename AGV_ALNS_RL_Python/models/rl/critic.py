"""
Critic network for RL-based operator selection

Value network that estimates expected return
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .gnn import GINWithGlobalFeatures


class CriticNetwork(nn.Module):
    """
    Critic network for estimating state value

    Architecture:
    - GIN encoder processes solution graph
    - Output layer predicts state value (expected return)
    """

    def __init__(
        self,
        node_input_dim: int = 8,
        global_input_dim: int = 4,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize critic network

        Args:
            node_input_dim: Node feature dimension
            global_input_dim: Global feature dimension
            hidden_dim: Hidden layer dimension
            num_gnn_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(CriticNetwork, self).__init__()

        # GNN encoder (shared architecture with actor)
        self.encoder = GINWithGlobalFeatures(
            node_input_dim=node_input_dim,
            global_input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            node_features: Node features (num_nodes, node_input_dim)
            edge_index: Edge indices (2, num_edges)
            global_features: Global features (batch_size, global_input_dim)
            batch: Batch assignment (num_nodes,)

        Returns:
            State values (batch_size, 1)
        """
        # Encode solution state
        embedding = self.encoder(node_features, edge_index, global_features, batch)

        # Predict value
        value = self.value_head(embedding)

        return value

    def get_value(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get state value estimate

        Returns:
            State value (batch_size,)
        """
        value = self.forward(node_features, edge_index, global_features, batch)
        return value.squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network

    Shares the GNN encoder between actor and critic for efficiency
    """

    def __init__(
        self,
        node_input_dim: int = 8,
        global_input_dim: int = 4,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_destroy_operators: int = 6,
        num_repair_operators: int = 9,
        dropout: float = 0.1,
        shared_encoder: bool = True
    ):
        """
        Initialize Actor-Critic

        Args:
            node_input_dim: Node feature dimension
            global_input_dim: Global feature dimension
            hidden_dim: Hidden layer dimension
            num_gnn_layers: Number of GNN layers
            num_destroy_operators: Number of destroy operators
            num_repair_operators: Number of repair operators
            dropout: Dropout rate
            shared_encoder: Whether to share encoder between actor and critic
        """
        super(ActorCritic, self).__init__()

        self.shared_encoder = shared_encoder

        if shared_encoder:
            # Shared GNN encoder
            self.encoder = GINWithGlobalFeatures(
                node_input_dim=node_input_dim,
                global_input_dim=global_input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout
            )

            # Actor heads
            self.destroy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_destroy_operators)
            )

            self.repair_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_repair_operators)
            )

            # Critic head
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        else:
            # Separate actor and critic
            from .actor import ActorNetwork
            self.actor = ActorNetwork(
                node_input_dim, global_input_dim, hidden_dim,
                num_gnn_layers, num_destroy_operators, num_repair_operators, dropout
            )
            self.critic = CriticNetwork(
                node_input_dim, global_input_dim, hidden_dim,
                num_gnn_layers, dropout
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for both actor and critic

        Returns:
            (destroy_logits, repair_logits, value)
        """
        if self.shared_encoder:
            # Shared encoder
            embedding = self.encoder(node_features, edge_index, global_features, batch)

            destroy_logits = self.destroy_head(embedding)
            repair_logits = self.repair_head(embedding)
            value = self.value_head(embedding)

        else:
            # Separate networks
            destroy_logits, repair_logits = self.actor(
                node_features, edge_index, global_features, batch
            )
            value = self.critic(
                node_features, edge_index, global_features, batch
            )

        return destroy_logits, repair_logits, value

    def get_action_and_value(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ):
        """
        Get action selection and value estimate

        Returns:
            (destroy_action, repair_action, value, destroy_log_prob, repair_log_prob)
        """
        destroy_logits, repair_logits, value = self.forward(
            node_features, edge_index, global_features, batch
        )

        # Sample actions
        from torch.distributions import Categorical

        destroy_probs = F.softmax(destroy_logits, dim=-1)
        repair_probs = F.softmax(repair_logits, dim=-1)

        destroy_dist = Categorical(destroy_probs)
        repair_dist = Categorical(repair_probs)

        if deterministic:
            destroy_action = torch.argmax(destroy_probs, dim=-1)
            repair_action = torch.argmax(repair_probs, dim=-1)
        else:
            destroy_action = destroy_dist.sample()
            repair_action = repair_dist.sample()

        destroy_log_prob = destroy_dist.log_prob(destroy_action)
        repair_log_prob = repair_dist.log_prob(repair_action)

        return (
            destroy_action.item(),
            repair_action.item(),
            value.squeeze(-1),
            destroy_log_prob,
            repair_log_prob
        )
