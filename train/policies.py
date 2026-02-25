"""Neural-network trainable policies for RLRLGym."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from rlrlgym.featurize import vectorize_observation

from .network_config import NetworkConfig


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _relu_grad(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def _tanh(x: float) -> float:
    return math.tanh(x)


def _tanh_grad(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t


@dataclass
class MLPQNetwork:
    input_dim: int
    hidden_layers: Sequence[int]
    output_dim: int
    activation: str = "relu"
    learning_rate: float = 0.003
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        dims = [self.input_dim] + list(self.hidden_layers) + [self.output_dim]
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []

        for din, dout in zip(dims[:-1], dims[1:]):
            scale = 1.0 / max(1.0, din)
            w = [[(self._rng.random() * 2.0 - 1.0) * scale for _ in range(din)] for _ in range(dout)]
            b = [0.0 for _ in range(dout)]
            self.weights.append(w)
            self.biases.append(b)

    def _activation(self, x: float) -> float:
        if self.activation == "tanh":
            return _tanh(x)
        return _relu(x)

    def _activation_grad(self, x: float) -> float:
        if self.activation == "tanh":
            return _tanh_grad(x)
        return _relu_grad(x)

    def forward(self, x: List[float]) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        activations: List[List[float]] = [x]
        pre_acts: List[List[float]] = []
        current = x

        for li, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = []
            for i in range(len(w)):
                s = b[i]
                for j in range(len(current)):
                    s += w[i][j] * current[j]
                z.append(s)
            pre_acts.append(z)

            if li < len(self.weights) - 1:
                current = [self._activation(v) for v in z]
            else:
                current = list(z)
            activations.append(current)

        return activations[-1], activations, pre_acts

    def train_step(self, x: List[float], action: int, target: float) -> float:
        q_values, activations, pre_acts = self.forward(x)
        pred = q_values[action]
        error = pred - target
        loss = 0.5 * error * error

        deltas: List[List[float]] = [
            [0.0 for _ in range(len(layer))] for layer in activations[1:]
        ]
        deltas[-1][action] = error

        for li in range(len(self.weights) - 2, -1, -1):
            for j in range(len(deltas[li])):
                g = 0.0
                for i in range(len(deltas[li + 1])):
                    g += self.weights[li + 1][i][j] * deltas[li + 1][i]
                deltas[li][j] = g * self._activation_grad(pre_acts[li][j])

        for li in range(len(self.weights)):
            inp = activations[li]
            for i in range(len(self.weights[li])):
                d = deltas[li][i]
                self.biases[li][i] -= self.learning_rate * d
                for j in range(len(inp)):
                    self.weights[li][i][j] -= self.learning_rate * d * inp[j]

        return loss

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_layers": list(self.hidden_layers),
            "output_dim": self.output_dim,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "biases": self.biases,
        }

    def parameter_count(self) -> int:
        weight_params = sum(len(row) for layer in self.weights for row in layer)
        bias_params = sum(len(layer) for layer in self.biases)
        return int(weight_params + bias_params)


@dataclass
class NeuralQPolicy:
    net_cfg: NetworkConfig
    action_min: int = 0
    action_max: int = 11
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.epsilon = self.net_cfg.epsilon_start
        self.network: MLPQNetwork | None = None

    @property
    def actions(self) -> List[int]:
        return list(range(self.action_min, self.action_max + 1))

    def _ensure_network(self, obs: Dict[str, object]) -> None:
        if self.network is not None:
            return
        x = vectorize_observation(obs)
        self.network = MLPQNetwork(
            input_dim=len(x),
            hidden_layers=self.net_cfg.hidden_layers,
            output_dim=len(self.actions),
            activation=self.net_cfg.activation,
            learning_rate=self.net_cfg.learning_rate,
            seed=self.seed,
        )

    def ensure_initialized(self, observation: Dict[str, object]) -> None:
        self._ensure_network(observation)

    def parameter_count(self) -> int:
        if self.network is None:
            return 0
        return self.network.parameter_count()

    def act(self, observation: Dict[str, object], training: bool = True) -> int:
        self._ensure_network(observation)
        assert self.network is not None

        if training and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)

        x = vectorize_observation(observation)
        q_values, _, _ = self.network.forward(x)
        best_idx = max(range(len(q_values)), key=lambda i: q_values[i])
        return self.actions[best_idx]

    def update(
        self,
        observation: Dict[str, object],
        action: int,
        reward: float,
        next_observation: Dict[str, object] | None,
        done: bool,
    ) -> float:
        self._ensure_network(observation)
        assert self.network is not None

        x = vectorize_observation(observation)
        action_idx = action - self.action_min

        if done or next_observation is None:
            target = reward
        else:
            self._ensure_network(next_observation)
            nx = vectorize_observation(next_observation)
            next_q, _, _ = self.network.forward(nx)
            target = reward + self.net_cfg.gamma * max(next_q)

        return self.network.train_step(x, action_idx, target)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.net_cfg.epsilon_end, self.epsilon * self.net_cfg.epsilon_decay)

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "epsilon": self.epsilon,
            "config": {
                "name": self.net_cfg.name,
                "hidden_layers": self.net_cfg.hidden_layers,
                "activation": self.net_cfg.activation,
                "learning_rate": self.net_cfg.learning_rate,
                "gamma": self.net_cfg.gamma,
                "epsilon_start": self.net_cfg.epsilon_start,
                "epsilon_end": self.net_cfg.epsilon_end,
                "epsilon_decay": self.net_cfg.epsilon_decay,
            },
        }
        if self.network is not None:
            payload["network"] = self.network.to_dict()
        return payload
