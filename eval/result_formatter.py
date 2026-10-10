"""Format tracking results to TAO JSON + majority vote."""

import json
from collections import Counter, defaultdict


class ResultFormatter:
    """Collect per-frame tracking results and export TAO JSON.

    track_id global uniqueness: maintained via global_offset.
    Each video's tracker starts from 0; after video ends,
    global_offset += tracker.next_id.
    """

    def __init__(self):
        self.results = []
        self.global_offset = 0

    def add_frame(self, boxes, scores, categories, track_ids,
                  image_id, video_id, sources=None):
        """Add one frame's tracking results.

        Args:
            boxes: (K, 4) xyxy pixel coords.
            scores: (K,).
            categories: (K,) global TAO category_id (1-indexed), written as-is.
            track_ids: (K,) local track_id from tracker.
            image_id: TAO image id.
            video_id: TAO video id.
            sources: optional (K,) array of {'det','track'} per prediction.
                When provided, each result entry includes 'source' field for
                downstream FP-decomposition diagnostics.
        """
        if hasattr(boxes, 'numpy'):
            boxes = boxes.numpy()
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()
        if hasattr(categories, 'numpy'):
            categories = categories.numpy()
        if hasattr(track_ids, 'numpy'):
            track_ids = track_ids.numpy()
        if sources is not None and hasattr(sources, 'numpy'):
            sources = sources.numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            entry = {
                'image_id': int(image_id),
                'category_id': int(categories[i]),
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # xyxy -> xywh
                'score': float(scores[i]),
                'video_id': int(video_id),
                'track_id': self.global_offset + int(track_ids[i]),
            }
            if sources is not None:
                src = sources[i]
                entry['source'] = (src.decode() if isinstance(src, bytes)
                                   else str(src))
            self.results.append(entry)

    def on_video_end(self, tracker_next_id):
        """Call after each video to update global track_id offset.

        tracker_next_id is already the next available ID, no +1 needed.
        """
        self.global_offset += tracker_next_id

    def finalize(self, output_path, use_majority_vote=True):
        """Apply majority vote and save to JSON.

        Returns:
            output_path: path to saved file.
        """
        results = self.results
        if use_majority_vote and len(results) > 0:
            results = majority_vote(results)
        with open(output_path, 'w') as f:
            json.dump(results, f)
        print(f'Saved {len(results)} track results to {output_path}')
        return output_path


def majority_vote(results):
    """For each track_id, replace category_id with the mode (most common).

    track_id is already globally unique, so no need to group by video_id.
    """
    groups = defaultdict(list)
    for r in results:
        groups[r['track_id']].append(r)

    out = []
    for items in groups.values():
        cats = [r['category_id'] for r in items]
        majority_cat = Counter(cats).most_common(1)[0][0]
        for r in items:
            entry = dict(r)
            entry['category_id'] = majority_cat
            out.append(entry)
    return out
