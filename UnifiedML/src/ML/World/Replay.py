# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import atexit
import random
from pathlib import PosixPath
from threading import Thread, Lock
from math import inf
import os

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch import multiprocessing as mp

from World.Memory import Memory, Batch
from World.Dataset import load_dataset, datums_as_batch, get_dataset_path, worker_init_fn, compute_stats
from minihydra import instantiate, open_yaml, Args


class Replay:
    def __init__(self, path='Replay/', batch_size=1, device='cpu', num_workers=0, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf,
                 save=False, mem_size=None,
                 partition_workers=False, with_replacement=False, done_episodes_only=None, shuffle=True,
                 fetch_per=1, prefetch_factor=3, pin_memory=False, pin_device_memory=False, rewrite_shape=None,
                 dataset=None, transform=None, frame_stack=1, nstep=None, discount=1, agent_specs=None):

        self.device = device
        self.offline = offline
        self.epoch = 1
        self.nstep = nstep or 0  # Future steps to compute cumulative reward from
        self.stream = stream

        self.begin_flag = Flag()  # Wait until first call to sample before initial fetch

        if self.stream:
            return

        self.trajectory_flag = Flag()  # Tell worker to include experience trajectories as well

        if self.offline:
            self.begin_flag.set()

        # CPU workers
        num_workers = max(1, min(num_workers, os.cpu_count()))

        self.memory = Memory(num_workers=num_workers,
                             gpu_capacity=gpu_capacity,
                             pinned_capacity=pinned_capacity,
                             tensor_ram_capacity=tensor_ram_capacity,
                             ram_capacity=ram_capacity,
                             hd_capacity=hd_capacity)

        self.rewrite_shape = rewrite_shape  # For rewritable memory

        self.add_lock = Lock() if hd_capacity == inf else None  # For adding to memory in concurrency

        if isinstance(path, PosixPath):
            path = path.as_posix()  # TODO Probably better to use PosixPath universally

        dataset_config = dataset or Args(_target_=None)
        dataset_config['capacities'] = sum(self.memory.capacities)
        card = Args({'_target_': dataset_config}) if isinstance(dataset_config, str) else dataset_config
        # Perhaps if Online, include whether discrete -> continuous, since action shape changes in just that case

        # Optional specs that can be set based on data
        norm, standardize, obs_spec, action_spec = [getattr(agent_specs, spec, None)
                                                    for spec in ['norm', 'standardize', 'obs_spec', 'action_spec']]

        if dataset_config is not None and dataset_config._target_ is not None:
            # TODO Can system-lock w.r.t. save path if load_dataset is Dataset and Offline, then recompute load_dataset
            #   Only one process should save a previously non-existent memory at a time

            if offline:
                root = 'World/ReplayBuffer/Offline/'  # TODO can't control root?
                save_path = root + get_dataset_path(dataset_config, root)
            else:
                save_path = 'World/ReplayBuffer/Online/' + path

            # Memory save-path
            self.memory.set_save_path(save_path)

            # Pytorch Dataset or Memory path
            dataset = load_dataset('World/ReplayBuffer/Offline/', dataset_config) if offline else save_path

            # Fill Memory
            if isinstance(dataset, str):
                # Load Memory from path
                if os.path.exists(dataset):
                    self.memory.load(dataset, desc=f'Loading Replay from {dataset}')
                    card = open_yaml(dataset + 'card.yaml')  # TODO Doesn't exist for unsaved mmap'd Online
            else:
                batches = DataLoader(dataset, batch_size=mem_size or batch_size)

                # Add Dataset into Memory in batch-size chunks
                capacity = sum(self.memory.capacities)
                with tqdm(total=len(batches), desc='Loading Dataset into accelerated Memory...') as bar:
                    for i, data in enumerate(batches):
                        if len(self.memory) + len(data[-1]) > capacity:
                            bar.total = i
                            break
                        self.memory.add(datums_as_batch(data))
                        bar.update()
                    bar.refresh()

            if action_spec is not None and action_spec.discrete:
                if 'discrete_bins' not in action_spec or action_spec.discrete_bins is None:
                    action_spec['discrete_bins'] = len(card.classes)

                if 'high' not in action_spec or action_spec.high is None:
                    action_spec['high'] = len(card.classes) - 1

                if 'low' not in action_spec or action_spec.low is None:
                    action_spec['low'] = 0
        elif not offline:
            self.memory.set_save_path('World/ReplayBuffer/Online/' + path)

            # Load Memory from path
            dataset = 'World/ReplayBuffer/Online/' + path
            if os.path.exists(dataset):
                self.memory.load(dataset, desc=f'Loading Replay from {dataset}')
                card = open_yaml(dataset + 'card.yaml')

        card['capacities'] = sum(self.memory.capacities)

        # Save Online replay on terminate  TODO Delete if not save! Since no card, will try to load, crash on next run
        if not offline and save:
            self.memory.set_save_path('World/ReplayBuffer/Online/' + path)
            atexit.register(lambda: (self.memory.save(desc='Saving Replay Memory...', card=card),
                                     print('Successfully saved Replay Memory to', self.memory.save_path)))

        # TODO Add meta datum if meta_shape, and make sure add() also does - or make dynamic

        transform = instantiate(transform)

        # Sampler

        sampler = Sampler(data_source=self.memory,
                          shuffle=shuffle,
                          offline=offline,
                          with_replacement=with_replacement,
                          done_episodes_only=done_episodes_only)

        # Parallel worker for batch loading

        create_worker = Offline if offline else Online

        self.partitions = num_workers - 1 if partition_workers else False

        worker = create_worker(memory=self.memory,
                               fetch_per=None if offline else fetch_per,
                               sampler=None if offline else sampler,
                               partition_workers=partition_workers,
                               begin_flag=self.begin_flag,
                               transform=transform,
                               frame_stack=frame_stack or 1,
                               nstep=self.nstep,
                               trajectory_flag=self.trajectory_flag,
                               discount=discount)

        # Batch loading

        self.batches = torch.utils.data.DataLoader(dataset=worker,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   pin_memory=pin_memory and 'cuda' in device,  # or pin_device_memory
                                                   # pin_memory_device=device if pin_device_memory else '',
                                                   prefetch_factor=prefetch_factor if num_workers else 2,
                                                   # shuffle=shuffle and offline,  # Not compatible with Sampler
                                                   sampler=sampler if offline else None,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=bool(num_workers))

        # TODO Note: if loading agent with stats saved, this is unnecessary/redundant/perhaps the wrong stats
        # Fill in necessary obs_spec and action_spec stats from dataset
        if offline:
            if norm and ('low' not in obs_spec or 'high' not in obs_spec
                         or obs_spec.low is None or obs_spec.high is None) \
                    or standardize and ('mean' not in obs_spec or 'stddev' not in obs_spec
                                        or obs_spec.mean is None or obs_spec.stddev is None):
                if 'stats' not in card:  # TODO Store back to card if card already exists!
                    card['stats'] = compute_stats(self.batches)  # TODO Lock this after checking if card exists in path
                if obs_spec is not None:
                    obs_spec.update(card.stats)

        # Save to hard disk if Offline  TODO Lock this after checking if card exists in path
        if isinstance(dataset, Dataset) and offline:
            # if accelerate and self.memory.num_batches <= sum(self.memory.capacities[:-1]):
            #     root = 'World/ReplayBuffer/Offline/'
            #     self.memory.set_save_path(root + get_dataset_path(dataset_config, root))
            #     save = True
            if save or self.memory.num_batches > sum(self.memory.capacities[:-1]):  # Until save-delete check
                self.memory.save(desc='Memory-mapping Dataset for training acceleration and future re-use. '
                                      'This only has to be done once', card=card)
                print('Successfully saved Replay Memory to', self.memory.save_path)

        # Replay

        self._replay = None

    # Allows iteration via "next" (e.g. batch = next(replay))
    def __next__(self):
        if not self.begin_flag:
            self.begin_flag.set()

        if self.stream:
            # Environment streaming
            sample = self.stream
        else:
            # Replay sampling
            try:
                sample = next(self.replay)
            except StopIteration:
                self.epoch += 1
                self._replay = None  # Reset iterator when depleted
                sample = next(self.replay)

        return Batch({key: torch.as_tensor(value).to(device=self.device, non_blocking=True)
                      for key, value in sample.items()})

    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)  # Recreates the iterator when exhausted
        return self._replay

    def include_trajectories(self):
        self.trajectory_flag.set()

    def add(self, trace):
        if trace is None:
            trace = []
        elif isinstance(trace, (Args, dict)):
            trace = [trace]

        for batch in trace:
            if self.stream:
                self.stream = batch  # For streaming directly from Environment  TODO N-step in {0, 1}
            else:
                def add():
                    with self.add_lock:
                        self.memory.add(batch)  # Add to memory

                # Don't thread if hd_capacity < inf,
                #  TODO fix asynchronous add-induced deletion conflicting with worker __getitem__ of deleted index
                # if self.add_lock is None:
                #     self.memory.add(batch)  # Add to memory
                # else:
                #     Thread(target=add).start()  # Threading
                #     TODO Does a Lock block its own process; should be fine since that's the purpose of Thread lock
                self.memory.add(batch)  # Add to memory

    def set_tape(self, shape):
        self.rewrite_shape = shape or [0]

    def writable_tape(self, batch, ind, step):
        assert isinstance(batch, (dict, Batch)), f'expected \'batch\' to be dict or Batch, got {type(batch)}.'
        self.memory.writable_tape(batch, ind, step)

    def __len__(self):
        # Infinite if stream, else num episodes in Memory
        return int(9e9) if self.stream else len(self.memory)


class Worker(Dataset):
    def __init__(self, memory, fetch_per, sampler, partition_workers, begin_flag, transform,
                 frame_stack, nstep, trajectory_flag, discount):
        self.memory = memory
        self.fetch_per = fetch_per
        self.partition_workers = partition_workers
        self.begin_flag = begin_flag

        self.sampler = None if sampler is None else OnlineSampler(sampler)
        self.samples_since_last_fetch = 0

        self.transform = transform

        self.frame_stack = frame_stack
        self.nstep = nstep
        self.trajectory_flag = trajectory_flag
        self.discount = discount

        self.initialized = False

    @property
    def worker(self):
        try:
            return torch.utils.data.get_worker_info().id
        except AttributeError:
            return 0

    def sample(self, index=-1, update=False):
        if not self.initialized:
            self.memory.set_worker(self.worker)
            self.initialized = True

        # Periodically update memory
        while self.fetch_per and not self.samples_since_last_fetch % self.fetch_per or update:
            self.memory.update()  # Can make Online only

            if len(self.memory) and self.begin_flag:
                break

        self.samples_since_last_fetch += 1

        if self.sampler is not None:
            index = next(self.sampler)

        # Sample index
        if index == -1 or index > len(self.memory) - 1:
            index = random.randint(0, len(self.memory) - 1)  # Random sample an episode
        else:
            index = int(index)  # If index is a shared tensor, pytorch can bug when returning

        # Each worker can round index to their nearest allocated reciprocal to reproduce DrQV2 divide
        if self.partition_workers:
            while index == 0 and self.worker != 0 or index != 0 and index % len(self.memory.queues) != self.worker:
                index = (index + 1) % len(self.memory)

        # Retrieve from Memory
        episode = self.memory[index]

        if update:
            nstep = bool(self.nstep)  # Allows dynamic nstep
        else:
            nstep = self.nstep  # But w/o step as input, models can't distinguish later episode steps

        if len(episode) < nstep + 1:  # Make sure at least one nstep is present if nstep
            #  TODO Note if partition workers and not enough seed steps, need to wait till replay has num workers len
            return self.sample(update=True)

        step = random.randint(0, len(episode) - 1 - nstep)  # Randomly sample experience in episode
        experience = Args(episode[step])

        # Frame stack / N-step
        experience = self.compute_RL(episode, experience, step)

        # Transform
        if self.transform is not None:
            experience.obs = self.transform(experience.obs)

        # Add metadata
        # TODO Don't store these unless needed
        # experience['episode_index'] = index
        # experience['episode_step'] = step
        if 'step' in experience:
            experience.pop('step')
        if 'done' in experience:
            experience.pop('done')

        for key in experience:  # TODO Move this adaptively in try-catch to collate converting first to int32
            if getattr(experience[key], 'dtype', None) in [torch.int64, torch.float64]:
                # For some reason, casting to int32 can throw collate_fn errors  TODO Lots of things do
                experience[key] = experience[key].to(torch.float32)  # Maybe b/c some ints aren't as_tensor'd
            # experience[key] = torch.as_tensor(experience[key], dtype=torch.float32)  # Ints just generally tend to crash
            # experience[key] = torch.as_tensor(experience[key], dtype=torch.float32).clone()
            # TODO In collate fn, just have a default tensor memory block to map everything to,
            #  maybe converts int64 to int32

        return experience.to_dict()

    def compute_RL(self, episode, experience, step):
        # TODO Just apply nstep and frame stack as transforms nstep, frame_stack, transform

        # Frame stack
        def frame_stack(traj, key, idx):
            frames = traj[max([0, idx + 1 - self.frame_stack]):idx + 1]
            for _ in range(self.frame_stack - idx - 1):  # If not enough frames, reuse the first
                frames = traj[:1] + frames
            frames = torch.concat([torch.as_tensor(frame[key])
                                   for frame in frames]).reshape(frames[0][key].shape[0] * self.frame_stack,
                                                                 *frames[0][key].shape[1:])
            return frames

        # Present
        if self.frame_stack > 1:
            experience.obs = frame_stack(episode, 'obs', step)  # Need experience as own dict/Batch for this

        # Future
        if self.nstep:
            # Transition
            experience.action = episode[step + 1].action

            traj_r = torch.as_tensor([float(experience.reward)
                                      for experience in episode[step + 1:step + self.nstep + 1]])

            experience['next_obs'] = frame_stack(episode, 'obs', step + len(traj_r))

            # Trajectory TODO
            if self.trajectory_flag:
                experience['traj_r'] = traj_r
                traj_o = np.concatenate([episode['obs'][max(0, idx - i):max(idx + self.nstep + 1 - i, self.nstep + 1)]
                                         for i in range(self.frame_stack - 1, -1, -1)], 1)  # Frame_stack
                traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
                if 'label' in experience:
                    traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(len(traj_r) + 1)
            experience.reward = np.dot(discounts[:-1], traj_r).astype('float32')
            # experience['discount'] = 0 if episode[step + len(traj_r)].done else discounts[-1].astype('float32')
            experience['discount'] = discounts[-1].astype('float32')  # TODO Use above
        else:
            experience['discount'] = 1  # TODO just add via collate, not here, or check if exists in Q_learning

        return experience

    def __len__(self):
        return len(self.memory)


class Offline(Worker, Dataset):
    def __getitem__(self, index):
        return self.sample(index)  # Retrieve a single experience by index


class Online(Worker, IterableDataset):
    def __iter__(self):
        while True:
            yield self.sample()  # Yields a single experience


# Quick parallel one-time flag
class Flag:
    def __init__(self):
        self.flag = torch.tensor(False, dtype=torch.bool).share_memory_()
        self._flag = False

    def set(self):
        self.flag[...] = self._flag = True

    def __bool__(self):
        if not self._flag:
            self._flag = bool(self.flag)
        return self._flag


# Allows Pytorch Dataset workers to read from a sampler non-redundantly in real-time
class OnlineSampler:
    def __init__(self, sampler):
        self.main_worker = os.getpid()

        self.index = torch.zeros([], dtype=torch.int64).share_memory_()  # Int64

        self.read_lock = mp.Lock()
        self.read_condition = torch.tensor(False, dtype=torch.bool).share_memory_()
        self.index_condition = torch.tensor(False, dtype=torch.bool).share_memory_()

        self.sampler = sampler
        self.iterator = None

        Thread(target=self.publish, daemon=True).start()

    # Sample index publisher
    def publish(self):
        assert os.getpid() == self.main_worker, 'Only main worker can feed sample indices.'

        while True:
            # Wait until read is called in a process
            if self.read_condition:
                if self.iterator is None:
                    self.iterator = iter(self.sampler)
                try:
                    self.index[...] = next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.sampler)
                    self.index[...] = next(self.iterator)

                # Notify that index has been updated  Since Iterable Dataset sequential, currently not needed
                self.read_condition[...] = False
                self.index_condition[...] = True

    def __iter__(self):
        yield from self

    def __next__(self):
        with self.read_lock:
            # Notify that read has been called
            self.read_condition[...] = True

            # Wait until index has been updated
            while True:
                if self.index_condition:
                    index = int(self.index)
                    self.index_condition[...] = False
                    return index


# class NullSampler:
#     def __iter__(self):
#         yield None
#
#     def __len__(self):
#         return 0


# Sampling w/o replacement of Offline or dynamically-growing Online distributions
class Sampler:
    def __init__(self, data_source, shuffle=True, offline=True, with_replacement=False, done_episodes_only=None):
        self.data_source = data_source
        self.shuffle = shuffle
        self.offline = offline
        self.with_replacement = with_replacement

        if not self.offline and not self.with_replacement:
            assert shuffle, 'Online Sampler doesn\'t support not shuffling, got shuffle=False.'

        # TODO Check last_episode.done and give Episode done attr
        # Whether to sample in-progress episodes. This is crucial for some reason for Online RL. Defaults False Offline
        self.done_episodes_only = done_episodes_only is None and not offline or done_episodes_only or False

        self.size = len(self.data_source) - self.done_episodes_only

        self.indices = []

    def __iter__(self):
        if self.with_replacement:
            yield random.randint(0, len(self) - self.done_episodes_only - 1)
        elif self.offline:
            yield from torch.randperm(self.size).tolist() if self.shuffle else list(range(self.size))
        else:
            size = len(self) - self.done_episodes_only
            if size > 0:
                if not len(self.indices):
                    self.indices = list(range(size))
                elif size > self.size:
                    self.indices.extend(list(range(self.size, size)))
                self.size = size
                sample = random.randint(0, len(self.indices) - 1)
                last = self.indices[-1]
                self.indices[-1] = self.indices[sample]
                self.indices[sample] = last
                yield self.indices.pop()  # Ordinarily removing an element is O(n). pop is O(1).
            else:
                yield random.randint(0, size - 1)

    def __len__(self):
        return self.size if self.offline else len(self.data_source)
