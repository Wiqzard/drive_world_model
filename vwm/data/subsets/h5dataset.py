import glob
import os
import random
from typing import Any, Dict, Optional

from PIL import Image
from torchvision import transforms
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class H5VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str = "/var/tmp/yoda3/train",
        target_width=1024,
        target_height=576,
        num_frames=25,
    ):
        self.data_root = data_root
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames

        assert (
            target_height % 64 == 0 and target_width % 64 == 0
        ), "Resize to integer multiple of 64"
        self.img_preprocessor = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
        )

        assert os.path.isdir(self.data_root)

        self.shard_paths = sorted(
            glob.glob(os.path.join(self.data_root, "*.hdf5"))
            + glob.glob(os.path.join(self.data_root, "*.h5"))
        )

        assert len(self.shard_paths) > 0, (
            "h5: Directory does not have any .hdf5 files! Dir: " + self.data_dir
        )

        self.shard_lengths = H5VideoDataset.check_shard_lengths(self.shard_paths)
        self.num_per_shard = self.shard_lengths[0]
        self.total_num = sum(self.shard_lengths)

        assert len(self.shard_paths) > 0, (
            "h5: Could not find .hdf5 files! Dir: "
            + self.data_dir
            + " ; len(self.shard_paths) = "
            + str(len(self.shard_paths))
        )

        self.num_of_shards = len(self.shard_paths)

        self.shards = [h5py.File(p, "r") for p in self.shard_paths]

    def build_data_dict(self, image_seq, sample_dict):
        # log_cond_aug = self.log_cond_aug_dist.sample()
        # cond_aug = torch.exp(log_cond_aug)
        cond_aug = torch.tensor([0.0])
        data_dict = {
            "img_seq": torch.stack(image_seq),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([7]),
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug,
        }
        return data_dict

    def preprocess_image(self, image):
        # take numpy image
        # image = Image.open(image_path)
        image = Image.fromarray(image)
        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = (ori_h - tmp_h) // 2
            bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))
        image = image.resize(
            (self.target_width, self.target_height), resample=Image.LANCZOS
        )
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.img_preprocessor(image)
        return image

    def __len__(self):
        return self.total_num

    def get_indices(self, idx):
        shard_idx = np.digitize(idx, np.cumsum(self.shard_lengths))
        idx_in_shard = str(idx - sum(self.shard_lengths[:shard_idx]))
        return shard_idx, idx_in_shard

    def __getitem__(self, index):
        idx = index % self.total_num
        shard_idx, idx_in_shard = self.get_indices(idx)

        shard = self.shards[shard_idx]
        num_frames = len(shard[idx_in_shard])

        start_idx = random.randint(0, num_frames - self.num_frames)
        image_seq = []
        for i in range(self.num_frames):
            image = shard[idx_in_shard][str(start_idx + i)][:]
            image = self.preprocess_image(image)
            image_seq.append(image)
        return self.build_data_dict(image_seq, None)

    @staticmethod
    def _get_num_in_shard(shard_p):
        print(f"\rh5: Opening {shard_p}... ", end="")
        try:
            with h5py.File(shard_p, "r") as f:
                num_per_shard = len(f["len"].keys())
        except Exception as e:
            print(f"h5: Could not open {shard_p}! Exception: {e}")
            num_per_shard = -1
        return num_per_shard

    @staticmethod
    def check_shard_lengths(file_paths):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_paths: list of .hdf5 files
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        shard_lengths = []
        print("Checking shard_lengths in", file_paths)
        for i, p in enumerate(file_paths):
            shard_lengths.append(H5VideoDataset._get_num_in_shard(p))
        return shard_lengths


if __name__ == "__main__":
    dataset = H5VideoDataset("/var/tmp/yoda3/train")
    datapoint = dataset[0]
    print(0)
