from torch.utils.data import Dataset, DataLoader
import os

class DenoisingDataset(Dataset):
    """Sound denoising dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.train_mels =[]
        train_packages =os.listdir(os.path.join(self.root_dir))
        for train_package in train_packages:
            package =os.path.join(self.root_dir,train_package)
            mel_files =os.listdir(package)
            for mel_file in mel_files:
                mel_file =os.path.join(package,mel_file)
                self.train_mels.append(mel_file)

    def __len__(self):
        return len(self.train_mels)

    def __getitem__(self, idx):
        return self.train_mels[idx]