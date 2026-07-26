from typing import Any
from pathlib import Path
import torchaudio
from src.datasets.base_dataset import BaseDataset



class ASVDataset(BaseDataset):

    def __init__(
            self,
            audio_dir: str,
            protocol_path: str,
            instance_transforms: dict | None = None,
            limit: int | None = None
    ):

        self.audio_dir = Path(audio_dir)
        self.index = self.create_index(protocol_path)

        super().__init__(index=self.index, instance_transforms=instance_transforms, limit=limit, shuffle_index=False)


    def create_index(self, part):
        index_path = Path(part)
        index = []
        with open(index_path, "r") as file:
            for line in file:
                author, audio_name, _, __, true_value = line.strip().split()
                label = int(true_value == "bonafide")
                true_path = self.audio_dir / f"{audio_name}.flac"
                index.append({"path": true_path.as_posix(), "key": audio_name, "label": label})
        return index

    def load_object(self, path):
        """
        Load object from disk.


        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """
        speech, sr = torchaudio.load(path)
        return speech

    def __getitem__(self, ind: int):
        instance = self._index[ind]

        return self.preprocess_data(
            {"data_object": self.load_object(instance["path"]), "labels": instance["label"], "key": instance["key"]})
