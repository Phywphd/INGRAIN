"""TAO dataset loader: organizes frames by video for sequential inference."""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class FrameInfo:
    image_id: int
    video_id: int
    frame_id: int
    img_path: str
    width: int
    height: int
    dataset_name: str


class TAODataset:
    """TAO dataset loader.

    Reads categories directly from annotation JSON (not external class files)
    to ensure category_id alignment with GT.
    """

    def __init__(self, ann_file: str, img_root: str):
        with open(ann_file) as f:
            self.data = json.load(f)

        # All 1203 categories
        self.categories: Dict[int, dict] = {
            c['id']: c for c in self.data['categories']
        }

        # Videos
        self.videos: Dict[int, dict] = {
            v['id']: v for v in self.data['videos']
        }

        # Group frames by video_id
        self.img_root = img_root
        self.frames_by_video: Dict[int, List[dict]] = defaultdict(list)
        for img in self.data['images']:
            self.frames_by_video[img['video_id']].append(img)

    def get_class_names_and_cat_ids(self) -> Tuple[List[str], List[int]]:
        """Return (class_names, cat_ids) for the full 1203-class vocabulary."""
        sorted_ids = sorted(self.categories.keys())
        names = [self.categories[cid]['name'] for cid in sorted_ids]
        return names, sorted_ids

    def get_base_novel_split(self) -> Tuple[Set[str], Set[str]]:
        """Return (base_class_names, novel_class_names).

        base = freq 'c' or 'f', novel = freq 'r'.
        """
        base = {c['name'] for c in self.categories.values()
                if c['frequency'] != 'r'}
        novel = {c['name'] for c in self.categories.values()
                 if c['frequency'] == 'r'}
        return base, novel

    def get_dataset_name(self, video_id: int) -> str:
        """Get sub-dataset name with fallback."""
        v = self.videos[video_id]
        ds = v.get('metadata', {}).get('dataset')
        if ds:
            return ds
        # Fallback: parse from video name (e.g. "val/BDD/xxx")
        name = v.get('name', '')
        parts = name.split('/')
        return parts[1] if len(parts) >= 2 else 'unknown'

    def iter_videos(self):
        """Iterate videos, yielding (video_info, sorted_frames)."""
        for vid_id in sorted(self.frames_by_video):
            video_info = self.videos[vid_id]
            # val GT carries 'frame_id'; TAO-test GT uses 'frame_index' — fall
            # back so both splits sort chronologically (val behaviour unchanged).
            frames = sorted(
                self.frames_by_video[vid_id],
                key=lambda x: x.get('frame_id', x.get('frame_index', 0)))
            yield video_info, frames

    def num_videos(self) -> int:
        return len(self.frames_by_video)

    def num_frames(self) -> int:
        return sum(len(v) for v in self.frames_by_video.values())
