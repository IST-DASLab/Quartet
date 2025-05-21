from copy import deepcopy
from pathlib import Path
import tempfile

import torch
import wandb

from .utils import eval


class WeightAverager:
    def __init__(
        self,
        model,
        horizon=100,
        interval=1,
        save_dir=None,
        device=None,
        dtype=torch.float32,
        count=0,
    ):
        super().__init__()
        self.device = device  # Where to keep avg model
        self.dtype = dtype  # Precision for accumulation (>= float32)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.module = deepcopy(model).to(dtype=self.dtype, device=device)

        assert horizon % interval == 0, "Interval should divide period"
        self.interval = interval
        self.horizon = horizon
        self.period = horizon // interval
        if save_dir is None:
            # Keep in tempdir
            self._tempdir = tempfile.TemporaryDirectory()
            self.save_dir = Path(self._tempdir.name)
        else:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.count = count
        # check if there are any checkpoints saved in the directory and set
        # num_saved to number of checkpoints with name <= count
        self.num_saved = len(
            [f for f in self.save_dir.iterdir() if f.is_file() and int(f.stem) <= count]
        )

    @torch.no_grad()
    def step(self, model, is_master_rank=True):
        # Update module with current state
        if self.count % self.interval == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                rate = 1 / ((self.count % self.horizon) // self.interval + 1)
                avg.copy_(torch.lerp(avg, curr, rate))

        self.count += 1

        if self.count % self.horizon == 0 and is_master_rank:
            torch.save(
                self.module.to().state_dict(),
                self.save_dir / f"{self.count}.pt",
            )
            self.num_saved += 1

    def get_latest_like(self, model):
        # Return model for latest completed period
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)

        # Assumes that we saved at a specific iteration, will fail otherwise
        count = self.count - self.count % self.horizon
        latest_path = self.save_dir / f"{count}.pt"
        map_and_load_state_dict(new_model, torch.load(latest_path))

        return new_model

    def sweep_horizon_like(self, model, max_num=None):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)
        avg_state = deepcopy(self.module.state_dict())
        if max_num is None:
            max_num = self.num_saved
        # Assumes all points exist
        for n in range(min(self.num_saved, max_num)):
            # Load state from the corresponding checkpoint
            count = self.count - self.count % self.horizon - n * self.horizon
            state = torch.load(self.save_dir / f"{count}.pt")

            # Update average state
            for key, avg in avg_state.items():
                new = state[key].to(dtype=avg.dtype, device=avg.device)
                rate = 1 / (n + 1)
                avg.copy_(torch.lerp(avg, new, rate))

            # Set new_model state and yield it
            map_and_load_state_dict(new_model, avg_state)
            yield ((n + 1) * self.horizon, new_model)


def map_and_load_state_dict(model, state_dict):
    for key, m_val in model.state_dict().items():
        for alias in (
            f"_orig_mod.{key}",
            f"_orig_mod.module.{key}",
        ):  # handle compiled / nested model
            if key not in state_dict and alias in state_dict:
                key = alias
                break
        s_val = state_dict[key]
        m_val.copy_(s_val.to(device=m_val.device, dtype=m_val.dtype))


def eval_wa(
    curr_iter,
    model,
    weight_averager,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    if weight_averager.num_saved == 0:
        return
    if not cfg.wa_sweep_horizon:
        val_reader.set_step(0)
        val_acc, val_loss, val_perplexity = eval(
            weight_averager.get_latest_like(model).eval(),
            val_reader,
            cfg.device,
            max_num_batches=(
                val_reader.num_batches()
                if curr_iter == cfg.iterations or full_eval
                else cfg.eval_batches
            ),
            ctx=type_ctx,
            cfg=cfg,
        )

        if cfg.wandb:
            if curr_iter == cfg.iterations or full_eval:
                logs = {
                    "iter": curr_iter,
                    "final-val/loss_wa": val_loss,
                    "final-val/perplexity_wa": val_perplexity,
                    "final-val/acc_wa": val_acc,
                }
            else:
                logs = {
                    "iter": curr_iter,
                    "val/loss_wa": val_loss,
                    "val/perplexity_wa": val_perplexity,
                    "val/acc_wa": val_acc,
                }
            wandb.log(logs)
        print(
            f">WA Eval: Iter={curr_iter} "
            f"val_loss={val_loss:.3f} "
            f"val_pp={val_perplexity:.3f} "
            f"val_acc={val_acc:3f}"
        )
    else:
        losses = []
        for horizon, avg_model in weight_averager.sweep_horizon_like(
            model, cfg.max_num_wa_sweeps
        ):
            avg_model.eval()
            val_reader.set_step(0)
            _, val_loss, _ = eval(
                avg_model,
                val_reader,
                cfg.device,
                max_num_batches=(
                    val_reader.num_batches()
                    if curr_iter == cfg.iterations or full_eval
                    else cfg.eval_batches
                ),
                ctx=type_ctx,
                cfg=cfg,
            )

            losses.append((val_loss, horizon))
        if len(losses) == 0:  # in case of none saved yet
            return
        best_loss, best_horizon = sorted(losses)[0]

        print(f"WA Eval: {[(h, f'{l:0.3e}') for (l,h) in losses]}")

        if cfg.wandb:
            if curr_iter == cfg.iterations or full_eval:
                logs = {
                    "iter": curr_iter,
                    "final-val/loss_wa": losses[0][0],
                    "final-val/perplexity_wa": 2.71828 ** losses[0][0],
                    "final-val/best_loss_wa": best_loss,
                    "final-val/best_perplexity_wa": 2.71828**best_loss,
                }
            else:
                logs = {
                    "iter": curr_iter,
                    "val/loss_wa": losses[0][0],
                    "val/perplexity_wa": 2.71828 ** losses[0][0],
                    "val/best_loss_wa": best_loss,
                    "val/best_perplexity_wa": 2.71828**best_loss,
                    "wa_best_horizon": best_horizon,
                }
            wandb.log(logs)


class ExponentialWeightAverager:
    def __init__(
        self,
        model,
        interval=1,
        decay=0.95,
        device=None,
        warmup=0,
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = device  # Where to keep avg model
        self.dtype = dtype  # Precision for accumulation (>= float32)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.module = deepcopy(model).to(dtype=self.dtype, device=device)

        self.interval = interval
        self.decay = decay
        self.num_saved = 0
        self.warmup = warmup
        self.count = 0

    @torch.no_grad()
    def step(self, model, is_master_rank=True):
        # Update module with current state

        if self.count < self.warmup:
            self.count += 1
            return

        if self.count == self.warmup:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                avg.copy_(curr)

        elif self.count % self.interval == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                avg.copy_(torch.lerp(avg, curr, 1 - self.decay))
            self.num_saved += 1

        self.count += 1

        # if self.count % self.horizon == 0 and is_master_rank:
        #     torch.save(
        #         self.module.to(dtype=torch.bfloat16).state_dict(),
        #         self.save_dir / f"{self.count}.pt",
        #     )
        #     self.num_saved += 1

    def get_latest_like(self, model):
        # Return model for latest completed period
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)

        map_and_load_state_dict(
            new_model, self.module.to(dtype=torch.bfloat16).state_dict()
        )

        return new_model


def eval_ema(
    curr_iter,
    model,
    ema,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        ema.get_latest_like(model).eval(),
        val_reader,
        cfg.device,
        max_num_batches=(
            val_reader.num_batches()
            if curr_iter == cfg.iterations or full_eval
            else cfg.eval_batches
        ),
        ctx=type_ctx,
        cfg=cfg,
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss_ema": val_loss,
                "final-val/perplexity_ema": val_perplexity,
                "final-val/acc_ema": val_acc,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss_ema": val_loss,
                "val/perplexity_ema": val_perplexity,
                "val/acc_ema": val_acc,
            }
        wandb.log(logs)
    print(
        f">EMA Eval: Iter={curr_iter} "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )
