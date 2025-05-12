import cv2
import os
from PIL import Image, ImageDraw
import numpy as np
import BboxToolkit as bt
from .bbox_utils import get_trans_mat, transform_bbox
from torchvision.transforms import ToPILImage
transform = ToPILImage()

pi = 3.141592


def visualize_with_obboxes(img, obboxes, labels, args, default_color='green'):
    """
    Visualize oriented bounding boxes on the image based on given arguments and save the visualization.

    Args:
        img (np.ndarray): Content of the image file.
        obboxes (np.ndarray): Array of obboxes with shape [N, 5] where 5 -> (x_ctr, y_ctr, w, h, angle).
        labels (np.ndarray): Array of labels for the obboxes.
        args (dict): Dictionary of visualization arguments.
        default_color (str): Default color for the bounding boxes.
    """

    # Ensure the save directory exists
    save_path = args.get('save_path', '.')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load class names if provided
    class_names = None
    if args.get('shown_names') and os.path.isfile(args['shown_names']):
        with open(args['shown_names'], 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    # Convert obboxes to the specified bbox type if needed
    shown_btype = args.get('shown_btype')
    if shown_btype:
        obboxes = bt.bbox2type(obboxes, shown_btype)

    # Filtering by score threshold if scores are provided
    score_thr = args.get('score_thr', 0.2)
    if obboxes.shape[1] == 6:  # Assuming scores are provided as the last column
        scores = obboxes[:, 5]
        valid_indices = scores > score_thr
        obboxes = obboxes[valid_indices, :5]  # Exclude scores from obboxes
        labels = labels[valid_indices] if labels is not None else None
        scores = scores[valid_indices]
    else:
        scores = None

    # Visualization parameters
    colors = args.get('colors', default_color)
    thickness = args.get('thickness', 2.0)
    text_off = args.get('text_off', False)
    font_size = args.get('font_size', 10)
    
    if labels is None:
        colors = default_color
        text_off = True
        thickness /= 2.

    # Call visualization function from BboxToolkit
    bt.imshow_bboxes(img, obboxes, labels=labels, scores=scores,
                     class_names=class_names, colors=colors, thickness=thickness,
                     with_text=not text_off, font_size=font_size, show=False,
                     wait_time=args.get('wait_time', 0), out_file=save_path)

vis_args = {
    "save_dir": "",
    "save_path": "",
    "shown_btype": None,
    "shown_names": 
    "tools/dataset/misc/vis_configs/dota2_0/short_names.txt",
    "score_thr": 0.3,
    "colors": 
    "tools/dataset/misc/vis_configs/dota2_0/colors.txt",
    "thickness": 2.5,
    "text_off": False,
    "font_size": 12,
    "wait_time": 0
}

# Visualization function
def visualize_images(imgs, img_metas, bboxes, labels, prefix, save_dir = 
                     "data/DOTA/show_data_trans", default_color='green'):
    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        # Convert tensor image to PIL Image for visualization
        pil_img = transform(img.detach().cpu()).convert("RGB")
        np_img = np.array(pil_img)
        
        config = dict(vis_args)
        # Prepare save directory and file name based on img_meta information
        filename = img_meta['filename']
        base_name = os.path.basename(filename).split('.')[0]  # Extract file name without extension
        save_dir = f"{save_dir}/{base_name}"
        os.makedirs(save_dir, exist_ok=True)
        config['save_dir'] = save_dir
        save_path = os.path.join(save_dir, f"{prefix}_{base_name}.png")
        
        # Update vis_args with dynamic save_path
        config['save_path'] = save_path
        
        # Convert bboxes and labels to np.ndarray if necessary
        obboxes_np = bboxes[i].detach().cpu().numpy()
        labels_np = labels[i].detach().cpu().numpy() if labels is not None else None
        
        # Call the visualization function
        visualize_with_obboxes(np_img, obboxes_np, labels_np, config, default_color)

def process_visualization(teacher_info, student_info, teacher_data, student_data, output_dir):
    M = get_trans_mat(teacher_info["transform_matrix"], student_info["transform_matrix"])
    
    pseudo_bboxes, valid_masks = transform_bbox(
        'obb', teacher_info["det_bboxes"],
        M,
        [meta["img_shape"] for meta in student_info["img_metas"]]
    )
    
    visualize_images(
        teacher_data["img"], teacher_data["img_metas"],
        [dbb[:, :6] for dbb in teacher_info["det_bboxes"]], 
        teacher_info["det_labels"], "teacher",
        output_dir
    )
    
    visualize_images(
        student_data["img"], student_data["img_metas"],
        [pbb[:, :6] for pbb in pseudo_bboxes], 
        [label[mask] for label, mask in zip(teacher_info["det_labels"], valid_masks)], 
        "student", output_dir
    )
    
    visualize_images(
        teacher_data["img"], teacher_data["img_metas"],
        teacher_data['gt_obboxes'], 
        teacher_data['gt_labels'], "gt_teacher",
        output_dir
    )
    
    student_gt_obboxes, valid_masks = transform_bbox(
        'obb', teacher_data['gt_obboxes'],
        M,
        [meta["img_shape"] for meta in student_info["img_metas"]],
    )
    
    visualize_images(
        student_data["img"], student_data["img_metas"],
        student_gt_obboxes, 
        [gt_label[mask] for gt_label, mask in zip(teacher_data['gt_labels'], valid_masks)],
        "gt_trans_student", output_dir
    )
    
    visualize_images(
        student_data["img"], student_data["img_metas"],
        student_data['gt_obboxes'], 
        student_data['gt_labels'],
        "gt_student", output_dir
    )

# Visualization function to draw specific points with a given radius on an image
def visualize_points(imgs, img_metas, points, prefix, 
                     save_dir="data/points_visualization", point_color='green', radius=4):
    """
    Visualize specific points with a given radius on images.

    Parameters:
    - imgs (list of torch.Tensor): List of tensors containing the image data (HxWxC).
    - img_metas (list of dict): Metadata associated with each image (e.g., filename).
    - points (list of torch.Tensor): List of point sets, each tensor with shape (N, 2), 
        where N is the number of points.
    - prefix (str): Prefix for the saved image files.
    - save_dir (str): Directory where the visualized images will be saved.
    - point_color (str): Color used for drawing points (default is 'red').
    - radius (int): Radius of the circles to draw around each point (default is 4).

    """
    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        # Convert tensor image to PIL Image for visualization
        pil_img = transform(img.detach().cpu()).convert("RGB")
        np_img = np.array(pil_img)
        
        # Prepare save directory and file name based on img_meta information
        filename = img_meta['filename']
        base_name = os.path.basename(filename).split('.')[0]  # Extract file name without extension
        save_dir = os.path.join(save_dir, base_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prefix}_{base_name}.png")
        
        # Convert point tensor to numpy array
        points_np = points[i].detach().cpu().numpy()
        
        # Draw points as solid circles on the image
        img_with_points = Image.fromarray(np_img)
        draw = ImageDraw.Draw(img_with_points)
        
        # Iterate over points and draw solid circles
        for point in points_np:
            x, y = point
            top_left = (x - radius, y - radius)
            bottom_right = (x + radius, y + radius)
            draw.ellipse([top_left, bottom_right], fill=point_color)

        # Save the image with drawn points
        img_with_points.save(save_path)

# Visualization function to draw both OBBs and points on the same image
def visualize_images_with_points(imgs, img_metas, bboxes, labels, points, prefix, 
                                 save_dir="data/DOTA/show_data_trans", 
                                 bbox_color='red', point_color='green', radius=4,
                                 score_thr=None):
    """
    Visualize oriented bounding boxes and specific points with a given radius on images.

    Parameters:
    - imgs (list of torch.Tensor): List of tensors containing the image data (HxWxC).
    - img_metas (list of dict): Metadata associated with each image (e.g., filename).
    - bboxes (list of torch.Tensor): List of bounding boxes for each image, each tensor with shape (N, 5 or 6).
    - labels (list of torch.Tensor): List of labels corresponding to the bboxes.
    - points (list of torch.Tensor): List of point sets, each tensor with shape (P, 2), 
        where P is the number of points.
    - prefix (str): Prefix for the saved image files.
    - save_dir (str): Directory where the visualized images will be saved.
    - bbox_color (str): Color used for drawing bounding boxes (default is 'red').
    - point_color (str): Color used for drawing points (default is 'green').
    - radius (int): Radius of the circles to draw around each point (default is 4).
    - score_thr (float): Threshold for filtering bounding boxes based on scores (default is None).

    """
    # Transform to convert Tensor to PIL Image for visualization
    transform = ToPILImage()

    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        # Convert tensor image to PIL Image for visualization
        pil_img = transform(img.detach().cpu()).convert("RGB")
        np_img = np.array(pil_img)
        
        config = dict(vis_args)  # Assuming `vis_args` contains other config options
        # Prepare save directory and file name based on img_meta information
        filename = img_meta['filename']
        base_name = os.path.basename(filename).split('.')[0]  # Extract file name without extension
        save_dir_path = f"{save_dir}/{base_name}"
        os.makedirs(save_dir_path, exist_ok=True)
        config['save_dir'] = save_dir_path
        save_path = os.path.join(save_dir_path, f"{prefix}_{base_name}.png")
        
        # Update vis_args with dynamic save_path
        config['save_path'] = save_path
        if score_thr is not None:
            config['score_thr'] = score_thr
        
        # Convert bboxes and labels to numpy arrays if necessary
        obboxes_np = bboxes[i].detach().cpu().numpy()
        labels_np = labels[i].detach().cpu().numpy() if labels is not None else None
        
        # Convert point tensor to numpy array
        points_np = points[i].detach().cpu().numpy()
        
        # Create an image for drawing using PIL
        img_with_annotations = Image.fromarray(np_img)
        draw = ImageDraw.Draw(img_with_annotations)

        # 1. Draw points as solid circles on the image using PIL
        for point in points_np:
            x, y = point
            top_left = (x - radius, y - radius)
            bottom_right = (x + radius, y + radius)
            draw.ellipse([top_left, bottom_right], fill=point_color)

        # Convert PIL image back to NumPy array before passing to visualize_with_obboxes
        np_img_with_annotations = np.array(img_with_annotations)

        # 2. Draw oriented bounding boxes (OBBs)
        visualize_with_obboxes(np_img_with_annotations, obboxes_np, labels_np, config, bbox_color)