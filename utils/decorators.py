import time

from config_code.config_classes import OptionsConfig
from utils.utils import get_wandb_project_name, initialize_wandb
import wandb
import torch
import glob


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def wandb_decorator(func):
    def wrapper(options: OptionsConfig, *args, **kwargs):
        assert type(options) == OptionsConfig, \
            ("First argument must be an OptionsConfig object. "
             "When using this decorator, the first argument must be the options object."
             "eg: @wandb_decorator\n"
             "def _main(options: OptionsConfig):"
             "    ...")

        if options.use_wandb:
            project_name, run_name = get_wandb_project_name(options)
            initialize_wandb(options, project_name, run_name)

        result = func(options, *args, **kwargs)

        if options.use_wandb:
            wandb.finish()
        return result

    return wrapper


# def profile_decorator(func):
#     def wrapper(options: OptionsConfig, *args, **kwargs):
#         assert type(options) == OptionsConfig, \
#             ("See wandb_decorator for explanation.")
#
#         if options.profile:
#             wait, warmup, active, repeat = 1, 1, 2, 1
#             schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
#             profiler = torch.profiler.profile(
#                 schedule=schedule,
#                 on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"), with_stack=False)
#
#             with profiler:
#                 profiler_callback = TorchTensorboardProfilerCallback(profiler)
#                 result = func(options, *args, **kwargs)
#
#                 profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
#                 profile_art.add_file(glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
#                 wandb.log_artifact(profile_art)
#
#     return wrapper
