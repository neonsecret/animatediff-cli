import os.path
import threading
import decord
import torch

import pandas as pd
import torchvision.transforms as transforms

from pytube import YouTube
from torch.utils.data import Dataset
from einops import rearrange
from transformers import CLIPTokenizer

decord.bridge.set_bridge('torch')


class YoutubeTuneAVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
        filename: str = "tmp.mp4",
        store_dir: str = "tmp/"
    ):
        df = pd.read_csv(csv_path, header=None)
        self.items = df.to_dict('records')

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.filename = os.path.join(store_dir, filename)

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(width, antialias=True),
            transforms.CenterCrop(width),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return len(self.items)

    def fetch_video(self, index, filename):
        video_idx = self.items[index][0]
        caption = self.items[index][3]

        YouTube(f'https://youtu.be/{video_idx}').streams.get_by_resolution("360p").download(filename=filename)

        return caption

    def __getitem__(self, index):
        filename = self.filename + str(threading.get_ident())
        # load and sample video frames
        text = self.fetch_video(index, filename)
        vr = decord.VideoReader(filename, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)

        pixel_values = video.permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        pixel_values = self.pixel_transforms(pixel_values)
        del vr

        data = {
            "pixel_values": pixel_values,
            "text": text
        }

        return data


if __name__ == '__main__':
    d = YoutubeTuneAVideoDataset("D:/datasets/test.csv")
    dataloader = torch.utils.data.DataLoader(d, batch_size=4, num_workers=16)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["prompt_ids"]))
