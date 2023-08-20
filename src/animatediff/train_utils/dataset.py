import decord
import pandas as pd
from pytube import YouTube

from torch.utils.data import Dataset
from einops import rearrange

decord.bridge.set_bridge('torch')


class YoutubeTuneAVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
        filename: str = "tmp.mp4",
        tokenizer=None
    ):
        df = pd.read_csv(csv_path, header=None)
        self.items = df.to_dict('records')
        self.prompt = prompt

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.filename = filename
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def fetch_video(self, index):
        video_idx = self.items[index][0]
        caption = self.items[index][3]

        YouTube(f'https://youtu.be/{video_idx}').streams.get_by_resolution("360p").download(filename=self.filename)

        return caption

    def __getitem__(self, index):
        # load and sample video frames
        caption = self.fetch_video(index)
        vr = decord.VideoReader(self.filename, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        prompt_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        data = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids
        }

        return data
