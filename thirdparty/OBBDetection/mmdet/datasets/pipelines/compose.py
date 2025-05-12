import os
import cv2
import numpy as np
import BboxToolkit as bt
from mmdet.datasets.pipelines.obb.misc import polymask2obb, visualize_with_obboxes, vis_args
import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES

@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms, visualize=False):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        self.visualize = visualize
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        if self.visualize:
            parent_folder = "/home/liyuqiu/RS-PCT/data/DOTA/show_data_augs"
            base_name = os.path.splitext(data['img_info']['filename'])[0]
            folder_path = os.path.join(parent_folder, base_name)
            os.makedirs(folder_path, exist_ok=True)
            vis_args['save_dir'] = folder_path
        for idx, t in enumerate(self.transforms):
            data = t(data)
            if data is None:
                return None
            
            if self.visualize:
                # Get the class name of the transform
                transform_class_name = type(t).__name__
                
                # Update vis_args for the transformed image save path
                vis_args['save_path'] = os.path.join(folder_path, 
                                                f'transformed_image_step_{idx}_{transform_class_name}.png')
                if 'gt_masks' in data and type(data['gt_masks']).__name__ != 'BitmapMasks':
                    # After applying each transform, visualize the image with the current state of GT obboxes
                    if type(data['img']).__name__ == 'DataContainer':
                        transformed_img = data['img'].data.clone().detach().numpy()
                    else:
                        transformed_img = data['img'].copy()
                    
                    if type(data['gt_masks']).__name__ == 'DataContainer':
                        transformed_masks = data['gt_masks'].data
                    else:
                        transformed_masks = data['gt_masks']
                    
                    transformed_obboxes = polymask2obb(transformed_masks)
                    if 'gt_labels' in data:
                        if type(data['gt_labels']).__name__ == 'DataContainer':
                            labels = data['gt_labels'].data.clone().detach().numpy()
                        else:
                            labels = data['gt_labels']
                        visualize_with_obboxes(transformed_img, transformed_obboxes, labels, vis_args)
                    if 'transform_matrix' in data:
                        if type(data['img']).__name__ == 'DataContainer':
                            matrix_img = data['img'].data.clone().detach().numpy()
                        else:
                            matrix_img = data['img'].copy()
                        # Process each bbox
                        for bbox in data['ann_info']['bboxes']:
                            points = np.array([[bbox[i], bbox[i+1]] for i in range(0, len(bbox), 2)])
                            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))]) # Convert to homogeneous coordinates
                            transformed_points = (data['transform_matrix'] @ points_homogeneous.T).T

                            # Assuming the transformation doesn't include rotation, scale, or shear that would invalidate using cv2.polylines
                            # If it does, further processing to correctly draw the OBBs might be required
                            transformed_points = transformed_points[:, :2].astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(matrix_img, [transformed_points], isClosed=True, color=(0, 255, 0), thickness=2)
                        # Save or display the image
                        save_path = os.path.join(folder_path, 
                                                f'matrix_step_{idx}_{transform_class_name}.png')
                        cv2.imwrite(save_path, matrix_img)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
