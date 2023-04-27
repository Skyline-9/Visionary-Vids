# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import argparse
import logging

import nncore
import torch
import torch._dynamo
from nncore.engine import Engine, comm, set_random_seed
from nncore.engine.hooks import TensorboardWriter
from nncore.nn import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file")
    parser.add_argument("--checkpoint", help="load a checkpoint")
    parser.add_argument("--resume", help="resume from a checkpoint")
    parser.add_argument("--seed", help="random seed", type=int)
    parser.add_argument("--eval", help="evaluation mode", action="store_true")
    parser.add_argument("--launcher", help="job launcher")
    parser.add_argument("--tensorboard", action="store_true", help="log to tensorboard")
    args = parser.parse_args()
    return args


def main():
    torch._dynamo.config.log_level = logging.ERROR

    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    launcher = comm.init_dist(launcher=args.launcher)

    if comm.is_main_process() and not args.eval:
        timestamp = nncore.get_timestamp()
        work_dir = nncore.join("work_dirs", nncore.pure_name(args.config))
        work_dir = nncore.mkdir(work_dir, modify_path=True)
        log_file = nncore.join(work_dir, "{}.log".format(timestamp))
    else:
        log_file = work_dir = None

    logger = nncore.get_logger(log_file=log_file)
    logger.info(f"Environment info:\n{nncore.collect_env_info()}")
    logger.info(f"Elastic launcher: {launcher}")
    logger.info(f"Config: {cfg.text}")

    seed = args.seed if args.seed is not None else cfg.get("seed")
    seed = set_random_seed(seed, deterministic=True)
    logger.info(f"Using random seed: {seed}")

    model = build_model(cfg.model, dist=bool(launcher))
    model = torch.compile(model)  # PyTorch 2 Optimization
    logger.info(f"Model architecture:\n{model.module}")

    engine = Engine(
        model,
        cfg.data,
        stages=cfg.stages,
        hooks=cfg.hooks,
        work_dir=work_dir,
        seed=seed,
    )

    if checkpoint := args.checkpoint:
        engine.load_checkpoint(checkpoint)
    elif checkpoint := args.resume:
        engine.resume(checkpoint)

    if args.tensorboard:
        TensorboardWriter(log_dir="logs", input_to_model=cfg.data)

    engine.launch(eval=args.eval)


if __name__ == "__main__":
    main()
