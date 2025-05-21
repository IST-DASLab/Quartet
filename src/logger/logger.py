from collections import defaultdict
import copy
from functools import wraps
from pathlib import Path
import pickle

import torch
import torch.nn.functional as F
import wandb


class DynamicsLogger:
    def __init__(self, model, optimizer, cfg, output_folder, wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.iteration = 0
        self.output_folder = output_folder
        self.wandb = wandb

        self.interval = None  # Set in step

        if isinstance(cfg["interval"], int):
            self.interval = cfg["interval"]
        elif isinstance(cfg["interval"], (tuple, list)):
            # Tuples of iter, interval with iter0 < iter1 < ...
            # [(iter0, interval0), (iter1, interval1), (iter2, interval2)]
            # At iterX change interval to intervalX
            self.interval = cfg["interval"][0][1]

        # TODO: Add default cfg here once tested
        self.cfg = copy.deepcopy(cfg)
        if self.cfg["disk_stats"] == "all":
            self.cfg["disk_stats"] = self.cfg["stats"]
        if self.cfg["wandb_stats"] == "all":
            self.cfg["wandb_stats"] = self.cfg["stats"]

        self.stats = defaultdict(lambda: defaultdict(list))
        self.reducers = defaultdict(lambda: "rms")
        self.reducers.update(
            {
                # Make sure any signed stats are not reduced with RMS!
                # 'layer_cos_gradient_angle': 'mean',
                # 'neuron/cos_gradient_angle': 'mean',
                # 'vector/values': 'mean',
                # 'vector/update_values': 'mean',
            }
        )

        # Wrap the step method of the optimizer
        self.optimizer.original_step = self.optimizer.step
        self.optimizer.step = self_preserving_overwrite(self.step).__get__(
            self.optimizer
        )

        self.wandb_setup_complete = False

    def step(self, *args, **kwargs):
        # Dictionaries keyed by parameter name
        # NOTE: Some may be direct references to model / optimizer (do not change in-place)
        pre_params = dict()
        pre_grads = dict()
        pre_states = dict()
        post_params = dict()
        post_states = dict()

        if isinstance(self.cfg["interval"], int):
            self.interval = self.cfg["interval"]
        elif isinstance(self.cfg["interval"], (tuple, list)):
            # Tuples of iter, interval with iter0 < iter1 < ...
            # [(iter0, interval0), (iter1, interval1), (iter2, interval2)]
            # At iterX change interval to intervalX
            idx = 0
            while (
                idx < len(self.cfg["interval"])
                and self.cfg["interval"][idx][0] <= self.iteration
            ):
                idx += 1
            self.interval = self.cfg["interval"][idx - 1][1]

        if "eps" in self.optimizer.defaults:
            eps = self.optimizer.defaults["eps"]
        else:
            eps = 1e-8

        if self.iteration % self.interval == 0:
            for name, param in self.model.named_parameters():
                pre_params[name] = param.clone().detach()
                if param.grad is not None:
                    pre_grads[name] = param.grad
                else:
                    pre_grads[name] = None

                pre_states[name] = copy.deepcopy(self.optimizer.state[param])

            self.optimizer.original_step(*args, **kwargs)  # Assuming no change to grads

            for name, param in self.model.named_parameters():
                post_params[name] = param.detach()
                post_states[name] = self.optimizer.state[param]

            self.log_statistics(
                pre_params, post_params, pre_grads, pre_states, post_states, eps
            )
        else:
            # Normal optimizer step, no logging
            self.optimizer.original_step(*args, **kwargs)

        self.iteration += 1

    @torch.no_grad()
    def log_statistics(
        self, pre_params, post_params, pre_grads, pre_states, post_states, eps
    ):
        requested_stats = set(self.cfg["stats"])

        if {"layer_norm", "neuron_norm"} & requested_stats:
            for name, param in pre_params.items():
                if param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                # Compute neuron norms (assume shape K x C x ...)
                neuron_norm = torch.linalg.vector_norm(param.flatten(1), dim=1)

                if "layer_norm" in requested_stats:
                    # This makes more sense with layernorm
                    # for BN rms_neuron_norm is what we predict (closely related)
                    layer_norm = torch.linalg.vector_norm(neuron_norm)
                    self.stats["layer_norm"][name].append(layer_norm)
                if "neuron_norm" in requested_stats:
                    self.stats["neuron_norm"][name].append(neuron_norm)

        if {"layer_grad_norm", "neuron_grad_norm"} & requested_stats:
            for name, grad in pre_grads.items():
                if grad.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                # Compute neuron norms (assume shape K x C x ...)
                neuron_grad_norm = torch.linalg.vector_norm(grad.flatten(1), dim=1)

                if "layer_grad_norm" in requested_stats:
                    grad_norm = torch.linalg.vector_norm(neuron_grad_norm)
                    self.stats["layer_grad_norm"][name].append(grad_norm)
                if "neuron_grad_norm" in requested_stats:
                    self.stats["neuron_grad_norm"][name].append(neuron_grad_norm)

        if {"layer_update_norm", "neuron_update_norm"} & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                post_param = post_params[name]
                diff = post_param - pre_param

                # Compute neuron norms (assume shape K x C x ...)
                neuron_update_norm = torch.linalg.vector_norm(diff.flatten(1), dim=1)

                if "layer_update_norm" in requested_stats:
                    layer_norm = torch.linalg.vector_norm(neuron_update_norm)
                    self.stats["layer_update_norm"][name].append(layer_norm)
                if "neuron_update_norm" in requested_stats:
                    self.stats["neuron_update_norm"][name].append(neuron_update_norm)

        if {"layer_relative_update", "neuron_relative_update"} & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                post_param = post_params[name]
                diff = post_param - pre_param

                if "layer_relative_update" in requested_stats:
                    layer_diff_norm = torch.linalg.vector_norm(diff)
                    layer_norm = torch.linalg.vector_norm(pre_param)
                    layer_relative_update = (layer_diff_norm + eps) / (layer_norm + eps)
                    self.stats["layer_relative_update"][name].append(
                        layer_relative_update
                    )

                if "neuron_relative_update" in requested_stats:
                    neuron_diff_norm = torch.linalg.vector_norm(diff.flatten(1), dim=1)
                    neuron_norm = torch.linalg.vector_norm(pre_param.flatten(1), dim=1)
                    neuron_relative_update = (neuron_diff_norm + eps) / (
                        neuron_norm + eps
                    )
                    self.stats["neuron_relative_update"][name].append(
                        neuron_relative_update
                    )

        if {"layer_angular_update", "neuron_angular_update"} & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                post_param = post_params[name]
                pre_param = (
                    pre_param.double()
                )  # There is a lot of noise for small angles

                if "layer_angular_update" in requested_stats:
                    cos = F.cosine_similarity(
                        pre_param.flatten(), post_param.flatten(), dim=0
                    )
                    angles = torch.acos(torch.clamp(cos, min=-1, max=1))
                    self.stats["layer_angular_update"][name].append(angles)

                if "neuron_angular_update" in requested_stats:
                    cos = F.cosine_similarity(
                        pre_param.flatten(1), post_param.flatten(1), dim=1
                    )
                    angles = torch.acos(torch.clamp(cos, min=-1, max=1))
                    self.stats["neuron_angular_update"][name].append(angles)

        if {"layer_grad_alignment", "neuron_grad_alignment"} & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                pre_grad = pre_grads[name]
                mean_layer_alignment = self.layer_cosine_sim(pre_grad, pre_param)
                self.stats["layer_grad_alignment"][name].append(mean_layer_alignment)

                mean_neuron_alignment = self.neuron_cosine_sim(pre_grad, pre_param)
                self.stats["neuron_grad_alignment"][name].append(mean_neuron_alignment)

        if {
            "layer_grad_velocity_alignment",
            "neuron_grad_velocity_alignment",
        } & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                # Adam and similar only
                if "exp_avg" not in pre_states[name]:
                    self.stats["layer_grad_velocity_alignment"][name].append(
                        torch.tensor(0).to(pre_param.device)
                    )
                    self.stats["neuron_grad_velocity_alignment"][name].append(
                        torch.zeros(pre_param.shape[0]).to(pre_param.device)
                    )
                    continue

                pre_grad = pre_grads[name]
                pre_state = pre_states[name]
                pre_m = pre_state["exp_avg"]

                mean_layer_alignment = self.layer_cosine_sim(pre_grad, pre_m)
                self.stats["layer_grad_velocity_alignment"][name].append(
                    mean_layer_alignment
                )

                mean_neuron_alignment = self.neuron_cosine_sim(pre_grad, pre_m)
                self.stats["neuron_grad_velocity_alignment"][name].append(
                    mean_neuron_alignment
                )

        # TODO: Log averages for scalar vectors
        if "scalar_rms" in requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() >= 2:
                    # Only scalars and scalar vectors
                    continue

                self.stats["scalar_rms"][name].append((param**2).mean().sqrt())

        if "scalar_update_rms" in requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() >= 2:
                    # Only scalars and scalar vectors
                    continue

                post_param = post_params[name]
                diff = post_param - pre_param

                self.stats["scalar_update_rms"][name].append((diff**2).mean().sqrt())

        if "scalar_grad_rms" in requested_stats:
            for name, grad in pre_grads.items():
                if grad.dim() >= 2:
                    # Only scalars and scalar vectors
                    continue

                self.stats["scalar_grad_rms"][name].append((grad**2).mean().sqrt())

        # Could add similar per elements histograms for the following:
        # scalar_value
        # scalar_update_value
        # scalar_grad_value

        # More obscure metrics below, only used in select experiments
        if {
            "layer_mean_second_grad_moment",
            "neuron_mean_second_grad_moment",
        } & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                post_state = post_states[name]
                post_v = post_state["exp_avg_sq"]

                if "layer_mean_second_grad_moment" in requested_stats:
                    mean_v = torch.mean(post_v)
                    self.stats["layer_mean_second_grad_moment"][name].append(mean_v)

                if "neuron_mean_second_grad_moment" in requested_stats:
                    mean_v = torch.mean(post_v.flatten(1), dim=1)
                    self.stats["neuron_mean_second_grad_moment"][name].append(mean_v)

        if {
            "layer_second_grad_moment_std_mean_ratio",
            "neuron_second_grad_moment_std_mean_ratio",
        } & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                post_state = post_states[name]
                post_v = post_state["exp_avg_sq"]

                v_neuron_mean = post_v.flatten(1).mean(dim=1)
                v_neuron_std = post_v.flatten(1).std(dim=1)
                neuron_std_mean_ratio = torch.div(v_neuron_std, v_neuron_mean)
                self.stats["neuron_second_grad_moment_std_mean_ratio"][name].append(
                    neuron_std_mean_ratio
                )

                v_layer_mean = post_v.flatten(0).mean(dim=0)
                v_layer_std = post_v.flatten(0).std(dim=0)
                layer_std_mean_ratio = torch.div(v_layer_std, v_layer_mean)
                self.stats["layer_second_grad_moment_std_mean_ratio"][name].append(
                    layer_std_mean_ratio
                )

        if {"layer_scaled_grad_norm", "neuron_scaled_grad_norm"} & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                pre_grad = pre_grads[name]
                post_state = post_states[name]
                post_v = post_state["exp_avg_sq"]

                scaled_grad = torch.div(pre_grad, (post_v.sqrt() + 1e-8))

                layer_scaled_grad_norm = self.layer_norm(scaled_grad)
                self.stats["layer_scaled_grad_norm"][name].append(
                    layer_scaled_grad_norm
                )

                neuron_scaled_grad_norm = self.neuron_norm(scaled_grad)
                self.stats["neuron_scaled_grad_norm"][name].append(
                    neuron_scaled_grad_norm
                )

        if {
            "layer_scaled_grad_wd_projection",
            "neuron_scaled_grad_wd_projection",
        } & requested_stats:
            for name, pre_param in pre_params.items():
                if pre_param.dim() < 2:
                    # Only higher dimensional weights (linear, conv etc)
                    continue

                pre_grad = pre_grads[name]
                post_state = post_states[name]
                post_v = post_state["exp_avg_sq"]

                scaled_grad = torch.div(pre_grad, (post_v.sqrt() + 1e-8))
                layer_scaled_grad_wd_projection = self.layer_gradient_wd_project(
                    scaled_grad, pre_param
                )
                self.stats["layer_scaled_grad_wd_projection"][name].append(
                    layer_scaled_grad_wd_projection
                )

                neuron_scaled_grad_wd_projection = self.neuron_gradient_wd_project(
                    scaled_grad, pre_param
                )
                self.stats["neuron_scaled_grad_wd_projection"][name].append(
                    neuron_scaled_grad_wd_projection
                )

        T_disk = self.cfg["disk_save_interval"] or 0
        T_wandb = self.cfg["wandb_interval"] or 0

        # Maybe log to disk
        if T_disk and (self.iteration + self.interval) % (T_disk * self.interval) == 0:
            self.log_to_disk()

        # Maybe log to wandb
        if (
            self.wandb
            and T_wandb
            and (self.iteration + self.interval) % (T_wandb * self.interval) == 0
        ):
            self.log_to_wandb()

    def layer_gradient_wd_project(self, g_t, w_t):
        norm = self.layer_norm(w_t)
        dot_prod = torch.sum(w_t.flatten() * g_t.flatten(), dim=0)
        projection = torch.div(dot_prod, norm * norm)
        return projection

    def neuron_gradient_wd_project(self, g_t, w_t):
        norm = self.neuron_norm(w_t)
        dot_prod = torch.sum(w_t.flatten(1) * g_t.flatten(1), dim=1)
        projection = torch.div(dot_prod, norm * norm)
        return projection

    def layer_cosine_sim(self, v1, v2):
        return F.cosine_similarity(v1.flatten(), v2.flatten(), dim=0)

    def neuron_cosine_sim(self, v1, v2):
        return F.cosine_similarity(v1.flatten(1), v2.flatten(1), dim=1)

    def layer_norm(self, v1):
        return torch.linalg.vector_norm(v1.flatten(), dim=0)

    def neuron_norm(self, v1):
        return torch.linalg.vector_norm(v1.flatten(1), dim=1)

    def log_to_disk(self, free_buffers=True):
        out_dict = dict()
        T_disk = self.cfg["disk_save_interval"]
        for stat_name in self.cfg["disk_stats"]:
            out_dict[stat_name] = dict()
            reducer = self.reducers[stat_name]

            for param_name, values in self.stats[stat_name].items():
                values = torch.stack(values[-T_disk:])
                if self.cfg["disk_max_channels"] > 0 and values.dim() > 1:
                    values = values[:, : self.cfg["disk_max_channels"]]
                if self.cfg["disk_downsample"] > 1:
                    assert T_disk % self.cfg["disk_downsample"] == 0
                    values = values.reshape(
                        (
                            T_disk // self.cfg["disk_downsample"],
                            self.cfg["disk_downsample"],
                            -1,
                        )
                    )
                    if reducer == "mean":
                        values = values.mean(dim=1)
                    elif reducer == "rms":
                        values = (values**2).mean(dim=1).sqrt()
                    elif reducer == "first":
                        values = values[:, 0]
                    else:
                        raise ValueError(f"Unknown {reducer=}")

                values = values.cpu()
                out_dict[stat_name][param_name] = values

        out_path = Path(self.output_folder) / "dynamics.pkl"
        with open(out_path, "ab") as fp:
            # Multiple dumps in a single file
            # https://stackoverflow.com/a/12762056
            pickle.dump(out_dict, fp)

        if free_buffers:
            self.free_buffers("disk")

    def log_to_wandb(self, free_buffers=True):
        # Assume stats are logged as a list of tensors for each stat
        # Reducer can be individual samples (i.e. the first) or mean

        out_dict = dict()
        T_wandb = self.cfg["wandb_interval"]
        for stat_name in self.cfg["wandb_stats"]:
            out_dict[stat_name] = dict()
            reducer = self.reducers[stat_name]

            for param_name, values in self.stats[stat_name].items():
                values = torch.stack(values[-T_wandb:])

                if reducer == "mean":
                    values = values.mean(dim=0)
                elif reducer == "global_mean":
                    values = values.mean(dim=0).mean()
                elif reducer == "rms":
                    values = (values**2).mean(dim=0).sqrt()
                elif reducer == "global_rms":
                    values = (values**2).mean(dim=0).sqrt().mean()
                elif reducer == "first":
                    values = values[0]
                else:
                    raise ValueError(f"Unknown {reducer=}")

                values = values.cpu().numpy()

                if values.size > 1:
                    values = wandb.Histogram(values)

                out_dict[f"{stat_name}/{param_name}"] = values

        if not self.wandb_setup_complete:
            # For whatever reason using globs at init doesn't work
            wandb.define_metric("iter")
            for stat in out_dict:
                wandb.define_metric(stat, step_metric="iter")
            self.wandb_setup_complete = True

        out_dict["iter"] = self.iteration - (T_wandb - 1) * self.interval
        wandb.log(
            data=out_dict,
            # step=self.iteration-(T_wandb-1)*self.interval
        )

        if free_buffers:
            self.free_buffers("wandb")

    def free_buffers(self, set_name="all"):
        # Delete old stat values that are no longer needed i.e. those that
        # have been logged by both wandb and to disk where appropriate

        if set_name == "all":
            self.stats.clear()
            return
        if set_name == "disk":
            main = "disk_stats"
            other = "wandb_stats"
        elif set_name == "wandb":
            main = "wandb_stats"
            other = "disk_stats"
        else:
            raise ValueError(f"Unknown {set_name=}")

        private_stats = set(self.cfg[main]) - set(self.cfg[other])
        for stat in private_stats:
            del self.stats[stat]

        T_disk = self.cfg["disk_save_interval"] or 0
        T_wandb = self.cfg["wandb_interval"] or 0
        buffer_size = max(T_disk, T_wandb)
        shared_stats = set(self.cfg[main]) & set(self.cfg[other])
        for stat_name in shared_stats:
            for param_name in self.stats[stat_name]:
                new_buffer = self.stats[stat_name][param_name][-buffer_size:]
                self.stats[stat_name][param_name] = new_buffer

    @staticmethod
    def load_stats(path):
        path = Path(path)

        log_fragments = []
        with open(path, "rb") as f:
            while True:
                try:
                    log_fragments.append(pickle.load(f))
                except EOFError:
                    break

        out_dict = dict()
        for stat_name in log_fragments[0]:
            stat_dict = {}
            for param_name in log_fragments[0][stat_name]:
                chunks = []
                for log_fragment in log_fragments:
                    chunks.append(log_fragment[stat_name][param_name])
                stat_dict[param_name] = torch.concatenate(chunks)
            out_dict[stat_name] = stat_dict
        return out_dict


def move_to_cpu(data, clone=False):
    def recurse(data):
        if isinstance(data, dict):
            return {k: recurse(v) for k, v in data.items()}
        if isinstance(data, list):
            return [recurse(v) for v in data]
        if isinstance(data, tuple):
            return tuple(recurse(v) for v in data)

        if isinstance(data, torch.Tensor):
            data = data.detach()
            if clone:
                data = data.clone()
            return data.to(device="cpu")
        else:
            # Others int, float, str, None etc
            if clone:
                return copy.deepcopy(data)  # Copy just in case
            else:
                return data

    return recurse(data)


# Bind this to the original object to preserve the self property
# E.g. obj.method = self_preserving_overwrite(some_func).__get__(obj)
def self_preserving_overwrite(method):
    @wraps(method)
    def _impl(inner_self, *args, **kwargs):
        return method(*args, **kwargs)

    return _impl
