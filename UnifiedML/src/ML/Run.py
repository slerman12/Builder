# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch.multiprocessing

from Hyperparams.minihydra import instantiate, get_args, interpolate  # minihydra conveniently and cleanly manages sys args
from Utils import init, MT, MP, save, load


@get_args(source='Hyperparams/args.yaml')  # Hyper-param arg files located in ./Hyperparams
def main(args):
    if args.multi_task:
        return MT.launch(args.multi_task)  # Handover to multi-task launcher

    # Set random seeds, device
    init(args)

    # Train, test environments
    env = instantiate(args.environment) if args.train_steps else None
    generalize = instantiate(args.environment, train=False, seed=args.seed + 1234)

    for arg in ('obs_spec', 'action_spec', 'evaluate_episodes'):
        if hasattr(generalize.env, arg):
            setattr(args, arg, getattr(generalize.env, arg))
    interpolate(args)  # Update args

    # Experience replay
    replay = instantiate(args.replay) if args.train_steps else None

    # Agent
    agent = load(args.load_path, args.device, args.agent) if args.load \
        else instantiate(args.agent).to(args.device)
    # replay.set_tape(getattr(agent, 'rewrite_shape', ()))  # Optional rewritable memory

    # Synchronize multi-task models (if exist)
    agent = MT.unify_agent_models(agent, args.agent, args.device, args.load and args.load_path)

    train_steps = args.train_steps + agent.step

    # Logger / Vlogger
    logger = instantiate(args.logger)
    vlogger = instantiate(args.vlogger) if args.log_media else None

    # Start
    converged = training = args.train_steps == 0
    while True:
        # Evaluate
        if converged or args.evaluate_per_steps and agent.step % args.evaluate_per_steps == 0:

            for _ in range(args.generate or args.evaluate_episodes):
                exp, logs, vlogs = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                      vlog=args.log_media)

                logger.log(logs, 'Eval', exp if converged else None)

            logger.dump_logs('Eval')

            if args.log_media:
                vlogger.dump(vlogs, f'{agent.step}')

        if args.plot_per_steps and (agent.step + 1) % args.plot_per_steps == 0 and not args.generate or converged:
            instantiate(args.plotting)

        if converged:
            break

        # Rollout
        experiences, logs, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True

        replay.add(experiences)

        if env.episode_done:
            if args.log_per_episodes and (agent.episode - 2 * replay.offline) % args.log_per_episodes == 0:
                logger.log(logs, 'Train' if training else 'Seed', dump=True)

        converged = agent.step >= train_steps
        training = training or agent.step > args.seed_steps and len(replay) >= args.num_workers or replay.offline

        # Train agent
        if training and (args.learn_per_steps and agent.step % args.learn_per_steps == 0 or converged):

            for _ in range(args.learn_steps_after if converged else args.learn_steps):
                logs = agent.learn(replay)  # Learn
                if args.mixed_precision:
                    MP.update()  # For training speedup via automatic mixed precision

                if args.log_per_episodes:
                    logger.log(logs, 'Train')

        if training and args.save_per_steps and agent.step % args.save_per_steps == 0 or (converged and args.save):
            save(args.save_path, agent, args.agent, 'frame', 'step', 'episode', 'epoch')

        if training and args.load_per_steps and agent.step % args.load_per_steps == 0:
            agent = load(args.load_path, args.device, args.agent, ['frame', 'step', 'episode', 'epoch'], True)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
