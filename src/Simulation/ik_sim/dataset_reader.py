# pyright: reportPrivateImportUsage=false

import logging
import os
from pathlib import Path

import requests
import huggingface_hub as hf
from huggingface_hub.utils._http import UniqueRequestIdAdapter

# Register 'List' as an alias for 'Sequence' before datasets loads any schema.
# Older recordings use "_type": "List" which newer datasets lib doesn't recognise.
from datasets.features import features as _hf_features
from datasets.features.features import Sequence as _Sequence
if "List" not in _hf_features._FEATURE_TYPES:
    _hf_features._FEATURE_TYPES["List"] = _Sequence

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# from billie_utils.features import robot_types

_BILLIE_DIR = os.environ["BILLIE_DIR"]
_LEROBOT_CACHE = f"{_BILLIE_DIR}/.cache/huggingface/lerobot"


class DatasetReader:
    def __init__(self, repo_id: str, episode: int = 0):
        self.repo_id = repo_id
        self.dataset = DatasetReader.load(repo_id, episode)
        print(f"fps: {self.dataset.fps},num_frames: {self.dataset.num_frames},num_episodes: {self.dataset.num_episodes}")

        robot_type = self.dataset.meta.robot_type
        # self.robot_type = robot_types.validate_robot_type(robot_type)
        # print(f"Dataset robot type: {self.robot_type}")

    def path_to_file(self, file: str) -> str:
        return f"{_LEROBOT_CACHE}/{self.repo_id}/{file}"

    def is_file_cached(self, file: str) -> bool:
        """
        Checks if there's a file in the local cache of the dataset.

        Args:
            file: The relative path to the file in the dataset.
        Returns:
            True if the file is present in the local cache, False otherwise.
        """
        file_path = self.path_to_file(file)
        logging.info(f"Checking if file {file} is cached under: {file_path}")
        return os.path.exists(file_path)

    @staticmethod
    def load(repo_id: str, episode: int):
        local_dir = f"{_LEROBOT_CACHE}/{repo_id}"
        if not os.path.exists(local_dir):
            print(
                f"Downloading dataset [{repo_id}](https://huggingface.co/datasets/{repo_id})..."
            )

            # We use HF_HUB_OFFLINE=1 in billie to avoid huggingface_hub trying to access the network
            # But here we need to download the dataset, so we temporarily disable the offline mode
            # by configuring a custom HTTP backend. This uses a private API of huggingface_hub, so
            # this is not good, but there seem to be no better way to do it fo§r now.
            def backend_factory() -> requests.Session:
                session = requests.Session()
                session.mount("http://", UniqueRequestIdAdapter())
                session.mount("https://", UniqueRequestIdAdapter())
                return session

            hf.utils.configure_http_backend(backend_factory)  # override HF_HUB_OFFLINE=1
            Path(local_dir).parent.mkdir(parents=True, exist_ok=True)
            hf.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision="v2.1",
                local_dir=local_dir,
            )
            hf.utils.configure_http_backend()  # restore HF_HUB_OFFLINE=1 behavior
        DatasetReader._patch_list_feature_type(local_dir)
        dataset = LeRobotDataset(repo_id=repo_id, episodes=[episode])
        return dataset

    @staticmethod
    def _patch_list_feature_type(local_dir: str) -> None:
        """Replace deprecated 'List' feature type with 'Sequence' in all info.json files."""
        import glob
        for path in glob.glob(f"{local_dir}/**/info.json", recursive=True):
            with open(path, 'r') as f:
                content = f.read()
            if '"_type": "List"' in content:
                with open(path, 'w') as f:
                    f.write(content.replace('"_type": "List"', '"_type": "Sequence"'))
                logging.info(f"Patched List→Sequence in {path}")
