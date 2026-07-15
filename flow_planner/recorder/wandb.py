from typing import Dict, Optional

import torch
import wandb

from flow_planner.recorder import RecorderBase


class WandBRecorder(RecorderBase):
    def __init__(
        self,
        project: str,
        name: str,
        save_dir: str,
        rank: int = 0,
        wandb_id: Optional[str] = None,
    ):
        super().__init__()
        self.rank = int(rank)
        self.run = None
        self.id = wandb_id

        if self.rank == 0:
            self.run = wandb.init(
                project=project,
                name=name,
                dir=save_dir,
                id=wandb_id,
                resume="allow" if wandb_id is not None else None,
            )
            self.id = self.run.id

    def _to_loggable(self, values: Dict):
        loggable = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            loggable[key] = value
        return loggable

    def record_loss(self, loss: Dict, step: int):
        if self.run is not None:
            wandb.log(self._to_loggable(loss), step=step)

    def record_metric(self, metrics: Dict, step: int):
        if self.run is not None:
            wandb.log(self._to_loggable(metrics), step=step)

    def close(self):
        if self.run is not None:
            wandb.finish()
