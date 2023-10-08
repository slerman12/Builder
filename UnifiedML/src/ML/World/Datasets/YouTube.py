# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from vidgear.gears import CamGear

import torch

from torch.utils.data import Dataset


"""
How to train a Bittle-robot-dog object detector

(1) Get videos of Bittle
(2) Make two files on Macula called bittle_yt_videos_train.txt & bittle_yt_videos_test.txt with each line a link
(3) Run tributaries with this script:
    python XuLabAdvanced.py sweep=../Sweeps/Bittle 
(4) python XuLabAdvanced.py sweep=../Sweeps/Bittle plot=true

Example uses 'cd tributaries/Sweeps/Examples', but any custom server may be defined, 
or just run the same hyperparams locally with 'ml' (tributaries is a different library; see Readme)
"""


class YouTube(Dataset):
    """
    YouTube videos
    """

    def __init__(self, url=None, file=None, resolution=None, fps=None):  # TODO Eventually, temporal outputs
        assert url or file, 'URL or URL-file parameter are required.'

        urls = []
        if file is not None:  # Can pass in file of URLs
            with open(file, 'r') as f:
                urls = list(f.readlines())
        if url is not None:  # And/or just a YouTube URL
            urls.append(url)

        options = {}

        # if resolution:
        #     options.update({"CAP_PROP_FRAME_WIDTH": resolution[0],
        #                     "CAP_PROP_FRAME_HEIGHT": resolution[1]})  # Resolution
        #
        #     self.obs_spec = {'shape': (3, *resolution)}  # TODO This doesn't seem to work

        if fps:
            options.update({"CAP_PROP_FPS": fps})  # Framerate

        for url in urls:  # Get frames  Note: videos get stored on RAM  TODO Pre-compute video length(s), iterate below
            print('Downloading YouTube video from URL:', url)

            video = CamGear(source=url, stream_mode=True, **options).start()

            self.frames = []
            while True:
                frame = video.read()

                if frame is None:
                    break

                self.frames.append(torch.as_tensor(frame).permute(2, 0, 1))

    def __getitem__(self, ind):
        return self.frames[ind]  # TODO Single-output currently only supported by stream=true

    def __len__(self):
        return len(self.frames)
