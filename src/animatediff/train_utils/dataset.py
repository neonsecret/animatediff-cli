import urllib.request

import decord
import pandas as pd

from torch.utils.data import Dataset
from einops import rearrange

decord.bridge.set_bridge('torch')


class TuneAVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        df = pd.read_csv(csv_path)
        self.items = df.to_dict('records')

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return len(self.items)

    def remove_text(self, filename):
        pass

    def fetch_video(self, index, filename="tmp.mp4"):
        url = self.items[index]["contentUrl"]
        caption = self.items[index]["name"]

        urllib.request.urlretrieve(url, filename)

        self.remove_text(filename)
        return filename, caption

    def __getitem__(self, index):
        # load and sample video frames
        video_path, caption = self.fetch_video(index)
        vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example
