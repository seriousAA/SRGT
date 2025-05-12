import os
import os.path as osp
import time
from collections import defaultdict
from functools import partial
from random import sample
from mmdet.utils import calculate_nproc_gpu_source, find_gpu_memory_allocation, find_best_gpu


import BboxToolkit as bt
import cv2
import mmcv
import numpy as np

import torch.multiprocessing as mp
from mmdet.core import eval_arb_map, eval_arb_recalls
from mmdet.ops.nms import nms
from mmdet.ops.nms_rotated import obb_nms, BT_nms
from ..builder import DATASETS
from ..custom import CustomDataset


@DATASETS.register_module()
class DOTADataset(CustomDataset):

    def __init__(self,
                 task,
                 fp_ratio=0,
                 save_ori=False,
                 **kwargs):
        assert task in ['Task1', 'Task2']
        self.task = task
        self.fp_ratio = fp_ratio
        self.save_ori = save_ori
        super(DOTADataset, self).__init__(**kwargs)

    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            cls.custom_classes = False
            return None

        cls.custom_classes = True
        return bt.get_classes(classes)

    def load_annotations(self, ann_file):
        split_config = osp.join(ann_file, 'split_config.json')
        self.split_info = mmcv.load(split_config)

        ori_annfile = osp.join(ann_file, 'ori_annfile.pkl')
        self.ori_infos = mmcv.load(ori_annfile)['content']

        patch_annfile = osp.join(ann_file, 'patch_annfile.pkl')
        patch_dict = mmcv.load(patch_annfile)
        cls, contents = patch_dict['cls'], patch_dict['content']
        self.ori_CLASSES = cls
        if self.CLASSES is None:
            self.CLASSES = cls

        if self.test_mode:
            return contents

        self.pp_infos = []
        self.fp_infos = []
        for content in contents:
            if content['ann']['bboxes'].size != 0:
                self.pp_infos.append(content)
            else:
                self.fp_infos.append(content)
        data_infos = self.add_random_fp()
        return data_infos

    def add_random_fp(self):
        if self.fp_ratio == 0 or self.filter_empty_gt:
            return self.pp_infos
        elif self.fp_ratio == 'all':
            return self.pp_infos + self.fp_infos
        else:
            num = min(self.fp_ratio*len(self.pp_infos), len(self.fp_infos))
            fp_infos = sample(self.fp_infos, k=int(num))
            return self.pp_infos + fp_infos

    def get_subset_by_classes(self):
        bt.change_cls_order(self.data_infos, self.ori_CLASSES, self.CLASSES)
        return self.data_infos

    def pre_pipeline(self, results):
        results['split_info'] = self.split_info
        results['cls'] = self.CLASSES
        super().pre_pipeline(results)

    def format_results(self,
                       results,
                       with_merge=True,
                       ign_scale_ranges=None,
                       iou_thr=0.5,
                       nproc=5,
                       save_dir=None,
                       non_cuda_parallel_merge=False,
                       **kwargs):
        nproc = min(nproc, os.cpu_count())
        task = self.task
        if mmcv.is_list_of(results, tuple):
            dets, segments = results
            if task == 'Task1':
                dets = _list_mask_2_obb(dets, segments)
        else:
            dets = results

        if not with_merge:
            results = [(data_info['id'], result)
                       for data_info, result in zip(self.data_infos, results)]
            if save_dir is not None:
                id_list, dets_list = zip(*results)
                bt.save_dota_submission(save_dir, id_list, dets_list, task, self.CLASSES)
            return results

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        if ign_scale_ranges is not None:
            assert len(ign_scale_ranges) == (len(self.split_info['rates']) *
                                             len(self.split_info['sizes']))
            split_sizes = []
            for rate in self.split_info['rates']:
                split_sizes += [int(size / rate) for size in self.split_info['sizes']]

        collector = defaultdict(list)
        for data_info, result in zip(self.data_infos, dets):
            if ign_scale_ranges is not None:
                img_scale = data_info['width']
                scale_ratio = np.array(split_sizes) / img_scale
                inds = np.argmin(abs(np.log(scale_ratio)))

                min_scale, max_scale = ign_scale_ranges[inds]
                min_scale = 0 if min_scale is None else min_scale
                max_scale = np.inf if max_scale is None else max_scale

            x_start, y_start = data_info['x_start'], data_info['y_start']
            new_result = []
            for i, dets in enumerate(result):
                if ign_scale_ranges is not None:
                    bbox_scales = np.sqrt(bt.bbox_areas(dets[:, :-1]))
                    valid_inds = (bbox_scales > min_scale) & (bbox_scales < max_scale)
                    dets = dets[valid_inds]
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                bboxes = bt.translate(bboxes, x_start, y_start)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(np.concatenate(
                    [labels, bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[data_info['ori_id']].append(new_result)

        threshold = 5e2
        merge_func = partial(
            _merge_func,
            CLASSES=self.CLASSES,
            iou_thr=iou_thr,
            task=task,
            threshold=threshold,
            non_cuda_parallel_merge=non_cuda_parallel_merge
        )
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_progress(
                merge_func, (collector.items(), len(collector)))
        else:
            print('Multiple processing')
            count_func = partial(
                _count_func,
                CLASSES=self.CLASSES,
                threshold=threshold
            )
            tasks = list(collector.items())
            if mp.get_start_method(allow_none=True) != 'spawn':
                # print(f"Current start method is {mp.get_start_method(allow_none=True)}, ")
                mp.set_start_method('spawn', force=True)
                # print(f"switch to {mp.get_start_method()} from now on.")
                
            print("Scan and sort the huge images based on the max number of "
                  "det bboxes among all categories")
            print(f"Threshold: {int(threshold)}")
            count_results = mmcv.track_parallel_progress(
                count_func, tasks, nproc, keep_order=True)
            
            easy_tasks = [task for task, flag in zip(tasks, count_results) if not flag]
            print(f"Use {nproc} subprocesses to handle the easy tasks using CPU.")
            easy_results = mmcv.track_parallel_progress(
                merge_func, easy_tasks, nproc)
            
            tough_tasks = [task for task, flag in zip(tasks, count_results) if flag]
            print("Accelerate the tough tasks by cuda using GPU.")
            if non_cuda_parallel_merge or calculate_nproc_gpu_source() < 2:
                tough_results = mmcv.track_progress(
                    merge_func, (tough_tasks, len(tough_tasks)))
            else:
                tough_results = mmcv.track_parallel_progress(
                    merge_func, tough_tasks, calculate_nproc_gpu_source())
                
            merged_results = easy_results + tough_results
        if save_dir is not None:
            id_list, dets_list = zip(*merged_results)
            if self.save_ori:
                bt.save_dota_submission_ori_classes(
                    save_dir, id_list, dets_list, task, self.CLASSES, 
                    ori_classes=self.ori_CLASSES)
            else:
                bt.save_dota_submission(save_dir, id_list, dets_list, task, self.CLASSES)

        stop_time = time.time()
        print('Used time: %.1f s' % (stop_time - start_time))
        return merged_results

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 with_merge=True,
                 ign_diff=True,
                 ign_scale_ranges=None,
                 save_dir=None,
                 merge_iou_thr=0.1,
                 use_07_metric=True,
                 scale_ranges=None,
                 eval_iou_thr=[0.5],
                 proposal_nums=(2000,),
                 nproc=5,
                 non_cuda_parallel_merge=False):
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        task = self.task

        eval_results = {}
        if metric == 'mAP':
            merged_results = self.format_results(
                results,
                nproc=nproc,
                with_merge=with_merge,
                ign_scale_ranges=ign_scale_ranges,
                iou_thr=merge_iou_thr,
                save_dir=save_dir,
                non_cuda_parallel_merge=non_cuda_parallel_merge)

            infos = self.ori_infos if with_merge else self.data_infos
            id_mapper = {ann['id']: i for i, ann in enumerate(infos)}
            det_results, annotations = [], []
            for k, v in merged_results:
                det_results.append(v)
                ann = infos[id_mapper[k]]['ann']
                gt_bboxes = ann['bboxes']
                gt_labels = ann['labels']
                diffs = ann.get(
                    'diffs', np.zeros((gt_bboxes.shape[0],), dtype=np.int64))

                if task == 'Task2':
                    gt_bboxes = bt.bbox2type(gt_bboxes, 'hbb')

                gt_ann = {}
                if ign_diff:
                    gt_ann['bboxes_ignore'] = gt_bboxes[diffs == 1]
                    gt_ann['labels_ignore'] = gt_labels[diffs == 1]
                    gt_bboxes = gt_bboxes[diffs == 0]
                    gt_labels = gt_labels[diffs == 0]
                gt_ann['bboxes'] = gt_bboxes
                gt_ann['labels'] = gt_labels
                annotations.append(gt_ann)

            print('\nStart calculate mAP!!!')
            print('Result is Only for reference,',
                  'final result is subject to DOTA_devkit')
            mean_ap, stats = eval_arb_map(
                det_results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=eval_iou_thr,
                use_07_metric=use_07_metric,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
            eval_results['results'] = []

            for i, stat in enumerate(stats):
                result = {'class': self.CLASSES[i]}
                for k, v in stat.items():
                    if k != 'precision':
                        if k == 'recall':
                            temp = np.array(v, ndmin=2)
                            result[k] = temp[:, -1].item() if temp.size > 0 else 0.
                        else:
                            result[k] = v
                eval_results['results'].append(result)
                
        elif metric == 'recall':
            assert mmcv.is_list_of(results, np.ndarray)
            gt_bboxes = []
            for info in self.data_infos:
                bboxes = info['ann']['bboxes']
                if ign_diff:
                    diffs = info['ann'].get(
                        'diffs', np.zeros((bboxes.shape[0],), dtype=np.int64))
                    bboxes = bboxes[diffs == 0]
                gt_bboxes.append(bboxes)
            if isinstance(eval_iou_thr, float):
                eval_iou_thr = [eval_iou_thr]
            recalls = eval_arb_recalls(
                gt_bboxes, results, True, proposal_nums, eval_iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(eval_iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

def _count_func(info, CLASSES, threshold=5e2):
    _, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)
    labels, dets = label_dets[:, 0], label_dets[:, 1:]
    for i in range(len(CLASSES)):
        cls_dets = dets[labels == i]

        if cls_dets.shape[0] > threshold:
            return True
    return False

def _merge_func(info, CLASSES, iou_thr, task, threshold=5e2, max_capacity=5e4, non_cuda_parallel_merge=False):
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)
    labels, dets = label_dets[:, 0], label_dets[:, 1:]
    nms_ops = bt.choice_by_type(nms, obb_nms, BT_nms, dets, with_score=True)

    big_img_results = []
    for i in range(len(CLASSES)):
        cls_dets = dets[labels == i]
        indices = np.random.choice(cls_dets.shape[0], min(cls_dets.shape[0], int(max_capacity)), replace=False)
        cls_dets = cls_dets[indices, :]
        
        if cls_dets.shape[0] > threshold:
            if non_cuda_parallel_merge:
                device_id = 0
            else:
                device_id = find_gpu_memory_allocation(os.getpid())
                device_id = device_id if device_id else find_best_gpu()
            if device_id is None:
                print("\nWarning: Too many bboxes in a single image:"
                    f"\nImage ID: {img_id}, Number of subpatches: {len(info[1])}, "
                    f"Category: {CLASSES[i]}, Number of detection bboxes: {cls_dets.shape[0]}")
            # if device_id is not None:
            #     print(f"Trying to use GPU device cuda:{device_id} to accelerate, "
            #         "it may take longer time than expected.")
            # else:
                print("Fail to find available GPU, have to use CPU for NMS ops, "
                    "the process will be significantly slower than expected!!!")
            nms_dets, _ = nms_ops(cls_dets, iou_thr, device_id=device_id)
        else:
            nms_dets, _ = nms_ops(cls_dets, iou_thr)

        if task == 'Task2':
            bboxes = bt.bbox2type(nms_dets[:, :-1], 'hbb')
            nms_dets = np.concatenate([bboxes, nms_dets[:, -1:]], axis=1)
        big_img_results.append(nms_dets)
    
    return img_id, big_img_results
    

def _list_mask_2_obb(dets, segments):
    new_dets = []
    for cls_dets, cls_segments in zip(dets, segments):
        new_cls_dets = []
        for ds, segs in zip(cls_dets, cls_segments):
            _, scores = ds[:, :-1], ds[:, -1]
            new_bboxes = []
            for seg in segs:
                try:
                    contours, _ = cv2.findContours(
                        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                except ValueError:
                    _, contours, _ = cv2.findContours(
                        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                max_contour = max(contours, key=len).reshape(1, -1)
                new_bboxes.append(bt.bbox2type(max_contour, 'obb'))

            new_bboxes = np.zeros((0, 5)) if not new_bboxes else \
                np.concatenate(new_bboxes, axis=0)
            new_cls_dets.append(
                np.concatenate([new_bboxes, scores[:, None]], axis=1))
        new_dets.append(new_cls_dets)
    return new_dets
