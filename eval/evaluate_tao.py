#!/usr/bin/env python3
"""TETA scoring helpers for INGRAIN evaluation."""

import os
import pickle
import sys

import numpy as np

from eval.config import TETA_PACKAGE_PATH


def compute_teta_base_novel(teta_res, base_class_names, novel_class_names):
    """Compute TETA separately for base and novel classes."""
    if 'COMBINED_SEQ' in teta_res:
        teta_res = teta_res['COMBINED_SEQ']

    frequent_teta = []
    rare_teta = []
    for key in teta_res:
        if key in base_class_names:
            frequent_teta.append(
                np.array(teta_res[key]['TETA'][50]).astype(float))
        elif key in novel_class_names:
            rare_teta.append(
                np.array(teta_res[key]['TETA'][50]).astype(float))

    header = ("{:<10} {:<10} {:<10} {:<10} {:<10} "
              "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}")
    print("\nBase and Novel classes performance")
    print(header.format(
        "TETA50:", "TETA", "LocS", "AssocS", "ClsS",
        "LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr"))

    base_mean = None
    if frequent_teta:
        base_mean = np.mean(np.stack(frequent_teta), axis=0)
        print("{:<10} ".format("Base"), end="")
        print(*["{:<10.3f}".format(num) for num in base_mean])
    else:
        print("No Base classes to evaluate!")

    novel_mean = None
    if rare_teta:
        novel_mean = np.mean(np.stack(rare_teta), axis=0)
        print("{:<10} ".format("Novel"), end="")
        print(*["{:<10.3f}".format(num) for num in novel_mean])
    else:
        print("No Novel classes to evaluate!")

    return base_mean, novel_mean


def run_teta_eval(ann_file, track_json, output_dir, dataset):
    """Run TETA evaluation and print the base/novel split."""
    sys.path.insert(0, TETA_PACKAGE_PATH)
    import teta

    eval_config = teta.config.get_default_eval_config()
    eval_config['PRINT_ONLY_COMBINED'] = True
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['OUTPUT_TEM_RAW_DATA'] = True
    eval_config['NUM_PARALLEL_CORES'] = int(
        os.environ.get('INGRAIN_TETA_CORES', '2'))

    dataset_config = teta.config.get_default_dataset_config()
    dataset_config['TRACKERS_TO_EVAL'] = ['INGRAIN']
    dataset_config['GT_FOLDER'] = ann_file
    dataset_config['OUTPUT_FOLDER'] = output_dir
    dataset_config['TRACKER_SUB_FOLDER'] = track_json

    print('\nRunning TETA evaluation...')
    evaluator = teta.Evaluator(eval_config)
    evaluator.evaluate(
        [teta.datasets.TAO(dataset_config)], [teta.metrics.TETA()])

    results_path = os.path.join(
        output_dir, 'INGRAIN', 'teta_summary_results.pth')
    if os.path.exists(results_path):
        teta_results = pickle.load(open(results_path, 'rb'))
        base_names, novel_names = dataset.get_base_novel_split()
        compute_teta_base_novel(teta_results, base_names, novel_names)
    else:
        print(f'Warning: TETA results not found at {results_path}')
