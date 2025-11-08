"""
Training script for RL-based operator selection

Uses REINFORCE (Policy Gradient) with baseline to train Actor-Critic
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import time
import os
from typing import List, Tuple
import argparse

from models.problem import AGVSolution, Task, AGV, ChargingStation
from models.alns.initial_solution import InitialSolutionGenerator
from models.rl import ActorCritic, GraphState, RLALNS, RLALNSConfig


class RLTrainer:
    """Trainer for RL-based operator selection"""

    def __init__(
        self,
        actor_critic: ActorCritic,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        device: str = 'cpu'
    ):
        """
        Initialize trainer

        Args:
            actor_critic: Actor-Critic model
            learning_rate: Learning rate
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
        """
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)

    def collect_episode(
        self,
        initial_solution: AGVSolution,
        max_steps: int = 50
    ) -> Tuple[List, List, List, float]:
        """
        Collect one episode of experience

        Args:
            initial_solution: Starting solution
            max_steps: Maximum steps per episode

        Returns:
            (states, actions, rewards, final_cost)
        """
        states = []
        destroy_actions = []
        repair_actions = []
        destroy_log_probs = []
        repair_log_probs = []
        values = []
        rewards = []

        current_solution = initial_solution.copy()
        current_cost = current_solution.calculate_objective()

        self.actor_critic.eval()

        for step in range(max_steps):
            # Convert solution to graph
            graph_state = GraphState(current_solution)
            graph_state.to(self.device)

            node_features, edge_index, _, global_features = graph_state.get_batch()

            # Get action and value
            with torch.no_grad():
                d_action, r_action, value, d_log_prob, r_log_prob = \
                    self.actor_critic.get_action_and_value(
                        node_features, edge_index, global_features
                    )

            # Store state and action
            states.append({
                'node_features': node_features.cpu(),
                'edge_index': edge_index.cpu(),
                'global_features': global_features.cpu()
            })
            destroy_actions.append(d_action)
            repair_actions.append(r_action)
            destroy_log_probs.append(d_log_prob)
            repair_log_probs.append(r_log_prob)
            values.append(value)

            # Apply operators
            from models.alns.operators import DESTROY_OPERATORS, REPAIR_OPERATORS

            destroy_names = list(DESTROY_OPERATORS.keys())
            repair_names = list(REPAIR_OPERATORS.keys())

            destroy_op = DESTROY_OPERATORS[destroy_names[d_action]]()
            repair_op = REPAIR_OPERATORS[repair_names[r_action]]()

            # Modify solution
            num_tasks = len(current_solution.tasks)
            num_remove = max(1, int(num_tasks * 0.2))

            new_solution = current_solution.copy()
            removed = destroy_op(new_solution, num_remove)
            repair_op(new_solution, removed)

            # Evaluate
            new_solution.update_metrics()
            new_cost = new_solution.calculate_objective()

            # Calculate reward (cost improvement)
            reward = max(0.0, current_cost - new_cost)  # Positive reward for improvement
            rewards.append(reward)

            # Update current solution
            current_solution = new_solution
            current_cost = new_cost

        return (
            states,
            (destroy_actions, repair_actions),
            (destroy_log_probs, repair_log_probs),
            values,
            rewards,
            current_cost
        )

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted returns (rewards-to-go)

        Args:
            rewards: List of rewards

        Returns:
            Discounted returns
        """
        returns = []
        R = 0.0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def train_step(
        self,
        states: List,
        actions: Tuple[List, List],
        old_log_probs: Tuple[List, List],
        returns: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Perform one training step

        Args:
            states: List of states
            actions: (destroy_actions, repair_actions)
            old_log_probs: (destroy_log_probs, repair_log_probs)
            returns: Discounted returns

        Returns:
            (actor_loss, critic_loss, entropy)
        """
        self.actor_critic.train()

        destroy_actions, repair_actions = actions
        old_destroy_log_probs, old_repair_log_probs = old_log_probs

        # Batch processing
        all_destroy_log_probs = []
        all_repair_log_probs = []
        all_values = []
        all_entropies = []

        for i, state in enumerate(states):
            node_features = state['node_features'].to(self.device)
            edge_index = state['edge_index'].to(self.device)
            global_features = state['global_features'].to(self.device)

            destroy_action = torch.tensor([destroy_actions[i]], device=self.device)
            repair_action = torch.tensor([repair_actions[i]], device=self.device)

            # Evaluate actions
            d_logits, r_logits, value = self.actor_critic(
                node_features, edge_index, global_features
            )

            # Compute log probs and entropy
            from torch.distributions import Categorical

            d_probs = torch.softmax(d_logits, dim=-1)
            r_probs = torch.softmax(r_logits, dim=-1)

            d_dist = Categorical(d_probs)
            r_dist = Categorical(r_probs)

            d_log_prob = d_dist.log_prob(destroy_action)
            r_log_prob = r_dist.log_prob(repair_action)

            entropy = d_dist.entropy() + r_dist.entropy()

            all_destroy_log_probs.append(d_log_prob)
            all_repair_log_probs.append(r_log_prob)
            all_values.append(value.squeeze())
            all_entropies.append(entropy)

        # Stack tensors
        destroy_log_probs = torch.stack(all_destroy_log_probs)
        repair_log_probs = torch.stack(all_repair_log_probs)
        values = torch.stack(all_values)
        entropies = torch.stack(all_entropies)

        # Compute advantages
        advantages = returns - values.detach()

        # Actor loss (policy gradient with baseline)
        actor_loss = -(destroy_log_probs * advantages).mean() - \
                     (repair_log_probs * advantages).mean()

        # Critic loss (value prediction)
        value_loss = nn.functional.smooth_l1_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropies.mean()

        # Total loss
        total_loss = (
            actor_loss +
            self.value_loss_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return actor_loss.item(), value_loss.item(), entropies.mean().item()

    def train(
        self,
        problem_generator,
        num_epochs: int = 200,
        episodes_per_epoch: int = 10,
        max_steps_per_episode: int = 50,
        log_dir: str = 'runs',
        save_dir: str = 'checkpoints',
        verbose: bool = True
    ):
        """
        Main training loop

        Args:
            problem_generator: Function that generates problem instances
            num_epochs: Number of training epochs
            episodes_per_epoch: Number of episodes per epoch
            max_steps_per_episode: Maximum steps per episode
            log_dir: Directory for tensorboard logs
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress
        """
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

        global_step = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_entropy = 0.0
            epoch_reward = 0.0
            epoch_improvement = 0.0

            for episode in range(episodes_per_epoch):
                # Generate problem instance
                initial_solution = problem_generator()

                initial_cost = initial_solution.calculate_objective()

                # Collect episode
                states, actions, log_probs, values, rewards, final_cost = \
                    self.collect_episode(initial_solution, max_steps_per_episode)

                # Compute returns
                returns = self.compute_returns(rewards)

                # Train
                actor_loss, critic_loss, entropy = self.train_step(
                    states, actions, log_probs, returns
                )

                # Track metrics
                episode_reward = sum(rewards)
                improvement = initial_cost - final_cost

                epoch_actor_loss += actor_loss
                epoch_critic_loss += critic_loss
                epoch_entropy += entropy
                epoch_reward += episode_reward
                epoch_improvement += improvement

                global_step += 1

                # Log to tensorboard
                writer.add_scalar('Episode/Reward', episode_reward, global_step)
                writer.add_scalar('Episode/Improvement', improvement, global_step)
                writer.add_scalar('Episode/FinalCost', final_cost, global_step)

            # Average metrics
            epoch_actor_loss /= episodes_per_epoch
            epoch_critic_loss /= episodes_per_epoch
            epoch_entropy /= episodes_per_epoch
            epoch_reward /= episodes_per_epoch
            epoch_improvement /= episodes_per_epoch

            # Learning rate schedule
            self.scheduler.step()

            # Log epoch metrics
            writer.add_scalar('Epoch/ActorLoss', epoch_actor_loss, epoch)
            writer.add_scalar('Epoch/CriticLoss', epoch_critic_loss, epoch)
            writer.add_scalar('Epoch/Entropy', epoch_entropy, epoch)
            writer.add_scalar('Epoch/AvgReward', epoch_reward, epoch)
            writer.add_scalar('Epoch/AvgImprovement', epoch_improvement, epoch)
            writer.add_scalar('Epoch/LearningRate', self.scheduler.get_last_lr()[0], epoch)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                epoch_time = time.time() - epoch_start_time
                print(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"ActorLoss={epoch_actor_loss:.4f}, "
                    f"CriticLoss={epoch_critic_loss:.4f}, "
                    f"Entropy={epoch_entropy:.4f}, "
                    f"AvgReward={epoch_reward:.2f}, "
                    f"AvgImprovement={epoch_improvement:.2f}, "
                    f"Time={epoch_time:.1f}s"
                )

            # Save checkpoint
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, checkpoint_path)
                if verbose:
                    print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pt')
        torch.save(self.actor_critic.state_dict(), final_path)
        if verbose:
            print(f"Training completed. Final model saved: {final_path}")

        writer.close()


def create_random_problem(
    num_tasks: int = 20,
    num_agvs: int = 3,
    num_stations: int = 2,
    grid_size: float = 100.0
) -> AGVSolution:
    """Generate a random problem instance"""

    # Generate random tasks
    tasks = []
    for i in range(num_tasks):
        location = (random.uniform(0, grid_size), random.uniform(0, grid_size))
        earliest_start = random.uniform(0, 500)
        latest_start = earliest_start + random.uniform(50, 200)
        duration = random.uniform(10, 50)
        energy_consumption = random.uniform(5, 20)

        tasks.append(Task(
            id=i,
            location=location,
            earliest_start=earliest_start,
            latest_start=latest_start,
            duration=duration,
            energy_consumption=energy_consumption
        ))

    # Generate charging stations
    stations = []
    for i in range(num_stations):
        location = (random.uniform(0, grid_size), random.uniform(0, grid_size))
        stations.append(ChargingStation(
            id=i,
            location=location,
            capacity=2,
            charging_rate=10.0
        ))

    # Generate AGVs
    agvs = []
    for i in range(num_agvs):
        agvs.append(AGV(
            id=i,
            battery_capacity=100.0,
            initial_soc=1.0,
            min_soc=0.2,
            speed=2.0
        ))

    # Create solution
    solution = AGVSolution(agvs, tasks, stations)

    # Generate initial solution
    solution = InitialSolutionGenerator.generate(solution, method='greedy_nearest')

    return solution


def main():
    parser = argparse.ArgumentParser(description='Train RL model for AGV ALNS')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per epoch')
    parser.add_argument('--steps', type=int, default=50, help='Steps per episode')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create model
    actor_critic = ActorCritic(
        node_input_dim=8,
        global_input_dim=4,
        hidden_dim=128,
        num_gnn_layers=3,
        num_destroy_operators=6,
        num_repair_operators=9,
        dropout=0.1,
        shared_encoder=True
    )

    # Create trainer
    trainer = RLTrainer(
        actor_critic=actor_critic,
        learning_rate=args.lr,
        device=args.device
    )

    # Train
    print("Starting training...")
    trainer.train(
        problem_generator=lambda: create_random_problem(num_tasks=20, num_agvs=3),
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        max_steps_per_episode=args.steps,
        verbose=True
    )


if __name__ == '__main__':
    main()
