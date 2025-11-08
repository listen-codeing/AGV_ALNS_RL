"""
Actor network for RL-based operator selection

Policy network that learns to select destroy/repair operators
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional

from .gnn import GINWithGlobalFeatures


class ActorNetwork(nn.Module):
    """
    Actor network for selecting ALNS operators

    Architecture:
    - GIN encoder processes solution graph
    - Output layers predict action probabilities
    - Actions: 6 destroy operators + 9 repair operators = 15 total
    """

    def __init__(
        self,
        node_input_dim: int = 8,
        global_input_dim: int = 4,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_destroy_operators: int = 6,
        num_repair_operators: int = 9,
        dropout: float = 0.1
    ):
        """
        Initialize actor network

        Args:
            node_input_dim: Node feature dimension
            global_input_dim: Global feature dimension
            hidden_dim: Hidden layer dimension
            num_gnn_layers: Number of GNN layers
            num_destroy_operators: Number of destroy operators
            num_repair_operators: Number of repair operators
            dropout: Dropout rate
        """
        super(ActorNetwork, self).__init__()

        self.num_destroy_operators = num_destroy_operators
        self.num_repair_operators = num_repair_operators
        self.num_actions = num_destroy_operators + num_repair_operators

        # GNN encoder
        self.encoder = GINWithGlobalFeatures(
            node_input_dim=node_input_dim,
            global_input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )

        # Action head for destroy operators
        self.destroy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_destroy_operators)
        )

        # Action head for repair operators
        self.repair_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_repair_operators)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            node_features: Node features (num_nodes, node_input_dim)
            edge_index: Edge indices (2, num_edges)
            global_features: Global features (batch_size, global_input_dim)
            batch: Batch assignment (num_nodes,)

        Returns:
            (destroy_logits, repair_logits)
        """
        # Encode solution state
        embedding = self.encoder(node_features, edge_index, global_features, batch)

        # Compute action logits
        destroy_logits = self.destroy_head(embedding)
        repair_logits = self.repair_head(embedding)

        return destroy_logits, repair_logits

    def select_action(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """
        Select destroy and repair operators

        Args:
            node_features: Node features
            edge_index: Edge indices
            global_features: Global features
            batch: Batch assignment
            deterministic: If True, use greedy selection; otherwise sample

        Returns:
            (destroy_action, repair_action, destroy_log_prob, repair_log_prob)
        """
        destroy_logits, repair_logits = self.forward(
            node_features, edge_index, global_features, batch
        )

        # Create distributions
        destroy_probs = F.softmax(destroy_logits, dim=-1)
        repair_probs = F.softmax(repair_logits, dim=-1)

        destroy_dist = Categorical(destroy_probs)
        repair_dist = Categorical(repair_probs)

        # Select actions
        if deterministic:
            destroy_action = torch.argmax(destroy_probs, dim=-1)
            repair_action = torch.argmax(repair_probs, dim=-1)
        else:
            destroy_action = destroy_dist.sample()
            repair_action = repair_dist.sample()

        # Get log probabilities
        destroy_log_prob = destroy_dist.log_prob(destroy_action)
        repair_log_prob = repair_dist.log_prob(repair_action)

        return (
            destroy_action.item(),
            repair_action.item(),
            destroy_log_prob,
            repair_log_prob
        )

    def evaluate_actions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        destroy_actions: torch.Tensor,
        repair_actions: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training

        Args:
            node_features: Node features
            edge_index: Edge indices
            global_features: Global features
            destroy_actions: Destroy actions taken
            repair_actions: Repair actions taken
            batch: Batch assignment

        Returns:
            (destroy_log_probs, repair_log_probs, entropy)
        """
        destroy_logits, repair_logits = self.forward(
            node_features, edge_index, global_features, batch
        )

        # Create distributions
        destroy_probs = F.softmax(destroy_logits, dim=-1)
        repair_probs = F.softmax(repair_logits, dim=-1)

        destroy_dist = Categorical(destroy_probs)
        repair_dist = Categorical(repair_probs)

        # Get log probabilities
        destroy_log_probs = destroy_dist.log_prob(destroy_actions)
        repair_log_probs = repair_dist.log_prob(repair_actions)

        # Calculate entropy for exploration bonus
        destroy_entropy = destroy_dist.entropy()
        repair_entropy = repair_dist.entropy()
        total_entropy = destroy_entropy + repair_entropy

        return destroy_log_probs, repair_log_probs, total_entropy

    def get_action_probs(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action probability distributions

        Returns:
            (destroy_probs, repair_probs)
        """
        destroy_logits, repair_logits = self.forward(
            node_features, edge_index, global_features, batch
        )

        destroy_probs = F.softmax(destroy_logits, dim=-1)
        repair_probs = F.softmax(repair_logits, dim=-1)

        return destroy_probs, repair_probs
