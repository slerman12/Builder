# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from vidgear.gears import CamGear

import torch

from torch.utils.data import Dataset


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

        if resolution:
            options.update({"CAP_PROP_FRAME_WIDTH": resolution[0],
                            "CAP_PROP_FRAME_HEIGHT": resolution[1]})  # Resolution

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

                if len(self.frames) > 10:
                    break

    def __getitem__(self, ind):
        return self.frames[ind]  # TODO Single-output currently only supported by stream=true

    def __len__(self):
        return len(self.frames)
