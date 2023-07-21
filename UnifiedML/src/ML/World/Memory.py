# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from math import inf
import atexit
import contextlib
import os
import warnings
import resource
from pathlib import Path

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import yaml

from tqdm import tqdm

import numpy as np

import torch

from minihydra import Args


class Memory:
    def __init__(self, save_path=None, num_workers=1, gpu_capacity=0, pinned_capacity=0,
                 tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf):
        self.id = id(self)
        self.worker = 0
        self.main_worker = os.getpid()

        self.capacities = [gpu_capacity, pinned_capacity, tensor_ram_capacity, ram_capacity, hd_capacity]

        self.save_path = save_path

        manager = mp.Manager()

        self.batches = manager.list()
        self.episode_trace = []
        self.episodes = []

        # Rewrite tape
        self.queues = [Queue()] + [mp.Queue() for _ in range(num_workers - 1)]

        # Counters
        self.num_batches_deleted = torch.zeros([], dtype=torch.int64).share_memory_()
        self.num_batches = self.num_experiences = self.num_experiences_mmapped = self.num_episodes_deleted = 0
        self.num_traces = 0

        atexit.register(self.cleanup)

        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)  # Shared memory can create a lot of file descr
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))  # Increase soft limit to hard limit

    def rewrite(self):  # TODO Thread w sync?
        # Before enforce_capacity changes index
        while not self.queue.empty():
            experience, episode, step = self.queue.get()

            self.episode(episode)[step] = experience

    def update(self):  # Maybe truly-shared list variable can tell workers when to do this / lock
        num_batches_deleted = self.num_batches_deleted.item()
        self.num_batches = max(self.num_batches, num_batches_deleted)

        for batch in self.batches[self.num_batches - num_batches_deleted:]:
            batch_size = batch.size()

            if not self.episode_trace:
                self.episodes.extend([Episode(self.episode_trace, i) for i in range(batch_size)])

            self.episode_trace.append(batch)

            self.num_batches += 1

            if batch['done']:
                self.episode_trace = []
                self.num_traces += 1

            self.num_experiences += batch_size
            self.enforce_capacity()  # Note: Last batch does enter RAM before capacity is enforced

    # TODO Be own thread https://stackoverflow.com/questions/14234547/threads-with-decorators
    def add(self, batch):
        assert self.main_worker == os.getpid(), 'Only main worker can send new batches.'
        # assert self.save_path is not None, 'Memory save_path must be set to add memories.'

        batch_size = Batch(batch).size()

        gpu = self.num_experiences + batch_size <= sum(self.capacities[:1])
        pinned = self.num_experiences + batch_size <= sum(self.capacities[:2])
        shared_tensor = self.num_experiences + batch_size <= sum(self.capacities[:3])
        shared = self.num_experiences + batch_size <= sum(self.capacities[:4])
        mmap = self.num_experiences + batch_size <= sum(self.capacities[:5])

        mode = 'gpu' if gpu else 'pinned' if pinned else 'shared_tensor' if shared_tensor \
            else 'shared' if shared else 'mmap' if mmap \
            else next(iter(self.episodes[0].batch(0).values())).mode  # Oldest batch

        if mode == 'mmap':
            assert self.save_path is not None, \
                f'Memory save_path must be set to add memory-mapped memories on hard disk.'

            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        batch = Batch({key: Mem(batch[key], f'{self.save_path}{self.num_batches}_{key}_{self.id}').to(mode)
                       for key in batch})  # TODO A meta key for special save_path

        self.batches.append(batch)
        self.update()  # TODO Pop updated from self.batches

    def writable_tape(self, batch, ind, step):  # TODO Should be its own thread
        assert self.main_worker == os.getpid(), 'Only main worker can send rewrites across the memory tape.'

        batch_size = 1

        for datum in batch.values():
            if getattr(datum, 'shape', None):
                batch_size = len(datum)

        experiences = [Batch({key: batch[key][i] if getattr(batch[key], 'shape', None) else batch[key]
                              for key in batch}) for i in range(batch_size)]

        for experience, ind, step in zip(experiences, ind, step):
            self.queues[int(ind % self.worker)].put((experience, ind, step))

        self.rewrite()

    def enforce_capacity(self):
        while self.num_experiences > sum(self.capacities):
            batch = self.episodes[0].batch(0)
            batch_size = batch.size()

            self.num_experiences -= batch_size

            if self.main_worker == os.getpid():
                self.num_batches_deleted[...] = self.num_batches_deleted + 1
                del self.batches[0]
                for i, mem in enumerate(batch.values()):
                    mem.delete()  # Delete oldest batch

            if next(iter(batch.values())).mode == 'mmap':
                self.num_experiences_mmapped -= batch_size

            del self.episodes[0][0]
            if not len(self.episodes[0]):
                del self.episodes[:batch_size]
                self.num_episodes_deleted += batch_size  # getitem ind = mem.index - self.num_episodes_deleted

    def trace(self, ind):
        return self.episodes[ind][0].episode_trace

    @property
    def traces(self):
        trace = None

        for i in range(len(self.episodes)):
            if self.trace(i) != trace:
                trace = self.trace(i)
                yield trace

    def episode(self, ind):
        return self.episodes[ind]

    def __getitem__(self, ind):
        return self.episode(ind)

    def __len__(self):
        return len(self.episodes)

    def cleanup(self):
        for batch in self.batches:
            for mem in batch.mems:
                with mem.cleanup():
                    pass

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_worker(self, worker):
        self.worker = worker

    @property
    def queue(self):
        return self.queues[self.worker]

    def load(self, load_path=None, desc='Loading Memory...'):
        assert self.main_worker == os.getpid(), 'Only main worker can call load.'

        if load_path is None:
            load_path = self.save_path

        mmap_paths = sorted(Path(load_path).glob('*_*_*_*_*'),
                            key=lambda path: int(path.stem.split('_', 1)[0]))

        batch = {}
        previous_num_batches = inf

        for i, mmap_path in enumerate(tqdm(mmap_paths, desc=desc)):
            num_batches, key, identifier, _ = mmap_path.stem.split('_', 3)

            if i == 0:
                self.id = identifier
                self.num_batches_deleted[...] = self.num_batches = int(num_batches)
            else:
                if self.id != identifier:
                    warnings.warn(f'Found Mems with multiple identifiers in load path {load_path}. Using id={self.id}.')
                    continue

                if int(num_batches) > previous_num_batches:
                    self.add(batch)  # TODO More efficient to add Batch of Mems and not, for example, doubly mmap
                    # TODO Also, make sure Mem is marked saved
                    batch = {}

            batch[key] = Mem(None, path=mmap_path).load()._mem

            previous_num_batches = int(num_batches)

        if batch:
            self.add(batch)

    # TODO Some kind of lock to mark crashes or based on presence of card
    def save(self, desc='Saving Memory...', card=None):
        assert self.main_worker == os.getpid(), 'Only main worker can call save.'
        assert self.save_path is not None, 'Memory save_path must be set to save memories.'

        if self.num_batches > 0:

            os.makedirs(self.save_path, exist_ok=True)

            for trace in tqdm(self.traces, desc=desc, total=self.num_traces, position=0):
                for batch in (tqdm(trace, desc='Saving Batches in Episode Trace.',
                                   position=1,  leave=None) if len(trace) > 1 else trace):
                    for mem in batch.mems:
                        mem.save()

            if card:
                if isinstance(card, Args):
                    card = card.to_dict()
                with open(self.save_path + 'card.yaml', 'w') as file:
                    yaml.dump(card, file)

    def saved(self, saved=True, desc='Setting saved flag in Mems...'):
        assert self.main_worker == os.getpid(), 'Only main worker can call saved.'
        assert self.save_path is not None, 'Memory save_path must be set to save memories.'

        for trace in tqdm(self.traces, desc=desc, total=self.num_batches):
            for batch in trace:
                for mem in batch.mems:
                    mem.saved = saved

class Queue:
    def __init__(self):
        self.queue = []

    def get(self):
        return self.queue.pop()

    def put(self, item):
        self.queue.append(item)

    def empty(self):
        return not len(self.queue)


class Episode:
    def __init__(self, episode_trace, ind):
        self.episode_trace = episode_trace
        self.ind = ind

    def batch(self, step):
        return self.episode_trace[step]

    def experience(self, step):
        return Experience(self.episode_trace, step, self.ind)

    def __setitem__(self, step, experience):
        stored_experience = self.experience(step)

        for key, datum in experience.items():
            stored_experience[key] = datum

    def __getitem__(self, step):
        if isinstance(step, slice):
            return [self.experience(s) for s in range(len(self))[step]]  # Slicing
        return self.experience(step)

    def __len__(self):
        return len(self.episode_trace)

    def __iter__(self):
        return (self.experience(i) for i in range(len(self)))

    def __delitem__(self, ind):
        self.episode_trace.pop(ind)


class Experience:
    def __init__(self, episode_trace, step, ind):
        self.episode_trace = episode_trace
        self.step = step
        self.ind = ind

    def datum(self, key):
        return self.episode_trace[self.step][key][self.ind]

    def keys(self):
        return self.episode_trace[self.step].keys()

    def values(self):
        return [self.datum(key) for key in self.keys()]

    def items(self):
        return zip(self.keys(), self.values())

    def __getitem__(self, key):
        return self.datum(key)

    def __getattr__(self, key):
        return self.datum(key)

    def __setitem__(self, key, experience):
        self.episode_trace[self.step][key][self.ind] = experience

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.episode_trace[self.step].keys())


class Batch(Args):
    @property
    def mems(self):  # An element can be Mem or datums
        yield from self.values()

    def size(self):
        for key, mem in self.items():
            try:
                if hasattr(mem, '__len__') and len(mem) > 1:
                    return len(mem)
            except TypeError:
                continue

        return 1


def as_numpy(data):
    return data if isinstance(data, np.ndarray) \
        else data.numpy() if isinstance(data, torch.Tensor) \
        else np.array(data)


# A multiprocessing Lock that can be stored and restored via __getstate__ and __setstate__
# Need to know num workers to make proper, to make sure all others False, otherwise coin toss
class Lock:
    def __init__(self):
        self.Lock = torch.tensor(False, dtype=torch.bool).share_memory_()

    def acquire(self):
        while self.Lock:
            pass
        self.Lock[...] = True

    def release(self):
        self.Lock[...] = False


class Mem:
    def __init__(self, mem, path=None):
        self._mem = None if mem is None else as_numpy(mem)
        self.path = str(path)
        self.saved = False

        self.mode = None if mem is None else 'ndarray'

        if mem is None:
            self.shape, self.dtype = (), None
        else:
            self.shape = tuple(self._mem.shape)
            self.dtype = self._mem.dtype
            self.path += '_' + str(self.shape) + '_' + self.dtype.name

        self.name = None

        self.Lock = Lock()

        self.main_worker = os.getpid()

    @contextlib.contextmanager
    def mem(self):
        if self.mode == 'shared':  # TODO Same for mmap! mmap.close(); don't store
            self.Lock.acquire()  # In case of two processes allocating to the same file descriptor, rare edge case
            shm = SharedMemory(name=self.name)
            self.Lock.release()
            yield np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            shm.close()
        else:
            yield self._mem

    def __getstate__(self):
        return self.path, self.saved, self.mode, self.main_worker, self.shape, self.dtype, self.name, self.Lock, \
            self._mem if self.mode in ('pinned', 'shared_tensor', 'gpu') else None

    def __setstate__(self, state):
        self.path, self.saved, self.mode, self.main_worker, self.shape, self.dtype, self.name, self.Lock, mem = state
        self._mem = np.memmap(self.path, self.dtype, 'r+', shape=self.shape) if self.mode == 'mmap' else mem

    def __getitem__(self, ind):
        with self.mem() as mem:
            mem = mem[ind] if self.shape else mem
            if self.mode == 'shared':
                mem = torch.as_tensor(mem).clone()  # shm gets closed if shared, so make copy
            return mem

    def __setitem__(self, ind, value):
        with self.mem() as mem:
            mem[ind if self.shape else ...] = value

            if self.mode == 'mmap':
                mem.flush()  # Write to hard disk

            self.saved = False

    @property
    def datums(self):
        with self.mem() as mem:
            # shm gets closed, so make copy if shared
            return torch.as_tensor(mem).clone() if self.mode == 'shared' else mem

    def tensor(self):
        with self.mem() as mem:
            return None if mem is None else torch.as_tensor(mem).to(non_blocking=True)

    def pinned(self):
        if self.mode != 'pinned':
            with self.cleanup():
                with self.mem() as mem:
                    if mem is not None:
                        self._mem = torch.as_tensor(mem).share_memory_().to(non_blocking=True).pin_memory()  # if cuda!
            self.mode = 'pinned'

        return self

    def shared_tensor(self):
        if self.mode != 'shared_tensor':
            with self.cleanup():
                with self.mem() as mem:
                    if mem is not None:
                        self._mem = torch.as_tensor(mem).share_memory_().to(non_blocking=True)
            self.mode = 'shared_tensor'

        return self

    def gpu(self):
        if self.mode != 'gpu':
            with self.cleanup():
                with self.mem() as mem:
                    if mem is not None:
                        self._mem = torch.as_tensor(mem).cuda(non_blocking=True)

            self.mode = 'gpu'

        return self

    def shared(self):
        if self.mode != 'shared':
            with self.cleanup():
                with self.mem() as mem:
                    if isinstance(mem, torch.Tensor):
                        mem = mem.numpy()
                    shm = SharedMemory(create=True, size=mem.nbytes)
                    self.name = shm.name
                    mem_ = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
                    mem_[...] = mem
                    self._mem = None
                    shm.close()

            self.mode = 'shared'

        return self

    def mmap(self):
        if self.mode != 'mmap':
            with self.cleanup():
                with self.mem() as mem:
                    if self.main_worker == os.getpid() and not self.saved:  # For online transitions
                        _mem = mem.copy() if isinstance(mem, np.memmap) \
                            else mem  # If already memory mapped, copy to prevent overwrite

                        self._mem = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)
                        self._mem[...] = _mem
                        self._mem.flush()  # Write to hard disk
                    else:
                        self._mem = np.memmap(self.path, self.dtype, 'r+', shape=self.shape)

            self.mode = 'mmap'
            self.saved = True

        return self

    def to(self, mode):
        if mode == 'pinned':
            return self.pinned()
        if mode == 'shared_tensor':
            return self.shared_tensor()
        if mode == 'shared':
            return self.shared()
        elif mode == 'mmap':
            return self.mmap()
        else:
            assert False, f'Mode "{mode}" not supported."'

    @contextlib.contextmanager
    def cleanup(self):
        yield
        if self.mode == 'shared':
            shm = SharedMemory(name=self.name)
            shm.close()
            if self.main_worker == os.getpid():
                shm.unlink()
            self.name = None

    def __bool__(self):
        with self.mem() as mem:
            return bool(mem)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def load(self):
        if not self.saved:
            _, shape, dtype = self.path.rsplit('_', 2)
            _mem = np.memmap(self.path, dtype, 'r+', shape=eval(shape))

            with self.mem() as mem:
                if mem is None:
                    self._mem = _mem
                    self.mode = 'mmap'
                    self.shape = eval(shape)
                    self.dtype = self._mem.dtype
                else:
                    if isinstance(mem, torch.Tensor):
                        _mem = torch.as_tensor(_mem)
                    mem[...] = _mem

            self.saved = True

        return self

    def save(self):
        if not self.saved:
            mmap = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)
            with self.mem() as mem:
                mmap[...] = mem
            mmap.flush()  # Write to hard disk
            self.saved = True

    def delete(self):
        with self.cleanup():
            if self.mode == 'mmap':
                if self.main_worker == os.getpid():
                    os.remove(self.path)

        self.saved = False


if torch.multiprocessing.current_process().name == 'MainProcess':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
