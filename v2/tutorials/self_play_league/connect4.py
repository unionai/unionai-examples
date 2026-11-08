"""Connect Four as batched tensors, two scripted opponents, and the policy-value network.

Every board is stored from the point of view of the player to move: +1 is their stone, -1 the
opponent's. After a move the board is negated, so the network always plays as +1 and one
network can play both sides.
"""

import torch
import torch.nn.functional as F
from torch import nn

ROWS, COLS = 6, 7

# Four 4x4 kernels, one per direction: horizontal, vertical, and the two diagonals.
_K = torch.zeros(4, 1, 4, 4)
_K[0, 0, 0, :] = 1
_K[1, 0, :, 0] = 1
_K[2, 0] = torch.eye(4)
_K[3, 0] = torch.eye(4).flip(1)


# {{docs-fragment env}}
class Boards:
    """A batch of Connect Four games, stepped together on one device."""

    def __init__(self, n: int, device: str = "cpu"):
        self.n, self.device = n, device
        self.kernels = _K.to(device)
        self.board = torch.zeros(n, ROWS, COLS, device=device)
        self.height = torch.zeros(n, COLS, dtype=torch.long, device=device)
        self.moves = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        mask = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        self.board[mask] = 0
        self.height[mask] = 0
        self.moves[mask] = 0

    def legal(self) -> torch.Tensor:
        return self.height < ROWS

    def wins(self, board: torch.Tensor) -> torch.Tensor:
        """True where the +1 player has four in a row."""
        hits = F.conv2d((board == 1).float().unsqueeze(1), self.kernels, padding=3)
        return (hits >= 4).flatten(1).any(1)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Play ``action`` for the player to move. Returns (mover won, game over)."""
        idx = torch.arange(self.n, device=self.device)
        row = ROWS - 1 - self.height[idx, action]
        self.board[idx, row, action] = 1
        self.height[idx, action] += 1
        self.moves += 1
        won = self.wins(self.board)
        over = won | (self.moves >= ROWS * COLS)
        self.board = -self.board  # the other player moves next
        return won, over
# {{/docs-fragment env}}


def winning_moves(boards: Boards, board: torch.Tensor) -> torch.Tensor:
    """For each game and column, would the +1 player win by playing there?"""
    out = torch.zeros(boards.n, COLS, dtype=torch.bool, device=boards.device)
    idx = torch.arange(boards.n, device=boards.device)
    legal = boards.legal()
    for c in range(COLS):
        test = board.clone()
        row = (ROWS - 1 - boards.height[:, c]).clamp(min=0)
        test[idx, row, c] = torch.where(legal[:, c], torch.ones_like(row, dtype=test.dtype), test[idx, row, c])
        out[:, c] = boards.wins(test) & legal[:, c]
    return out


# {{docs-fragment bots}}
def random_bot(boards: Boards, gen: torch.Generator | None = None) -> torch.Tensor:
    return torch.multinomial(boards.legal().float(), 1, generator=gen).squeeze(1)


CENTER = torch.tensor([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])


def tactical_bot(boards: Boards, gen: torch.Generator | None = None) -> torch.Tensor:
    """Win if it can, block if it must, otherwise play a random center-weighted move."""
    win = winning_moves(boards, boards.board)
    block = winning_moves(boards, -boards.board)
    weights = boards.legal().float() * CENTER.to(boards.device)
    weights = torch.where(block.any(1, keepdim=True), block.float(), weights)
    weights = torch.where(win.any(1, keepdim=True), win.float(), weights)
    return torch.multinomial(weights, 1, generator=gen).squeeze(1)


BOTS = {"random": random_bot, "tactical": tactical_bot}
# {{/docs-fragment bots}}


# {{docs-fragment net}}
class PolicyValueNet(nn.Module):
    """A small residual conv net: board in, move logits and a value in [-1, 1] out."""

    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1))
            for _ in range(blocks)
        )
        self.policy = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.Flatten(), nn.Linear(2 * ROWS * COLS, COLS))
        self.value = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Flatten(), nn.Linear(ROWS * COLS, 64), nn.ReLU(),
                                   nn.Linear(64, 1), nn.Tanh())

    def forward(self, board: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([(board == 1).float(), (board == -1).float()], 1)
        h = self.stem(x)
        for block in self.blocks:
            h = F.relu(h + block(h))
        logits = self.policy(h).masked_fill(~legal, -1e9)
        return logits, self.value(h).squeeze(1)
# {{/docs-fragment net}}


def net_bot(net: PolicyValueNet, greedy: bool = False, temperature: float = 1.0):
    """Wrap a network as an opponent function."""

    @torch.no_grad()
    def act(boards: Boards, gen: torch.Generator | None = None) -> torch.Tensor:
        logits, _ = net(boards.board, boards.legal())
        if greedy:
            return logits.argmax(1)
        return torch.multinomial(F.softmax(logits / temperature, 1), 1, generator=gen).squeeze(1)

    return act
