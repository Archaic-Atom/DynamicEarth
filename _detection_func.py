import os
import argparse

import torch
import numpy as np
from skimage.io import imread, imsave
from segment_anything.utils.amg import rle_to_mask
from segment_anything import sam_model_registry, sam_hq_model_registry

try:
    from .dynamic_earth.sam_ext import MaskProposal
    from .dynamic_earth.identifier.utils import identify, get_identifier
    from .dynamic_earth.comparator.bi_match import bitemporal_match
    from .dynamic_earth.utils import get_model_and_processor
except ImportError:
    from dynamic_earth.sam_ext import MaskProposal
    from dynamic_earth.identifier.utils import identify, get_identifier
    from dynamic_earth.comparator.bi_match import bitemporal_match
    from dynamic_earth.utils import get_model_and_processor


def merge_masks(change_masks, shape):
    """Merges individual change masks into a single change mask.

    Args:
        change_masks (list of np.array): List of individual change masks.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.array: Merged binary change mask.
    """
    if len(change_masks) == 0:
        return np.zeros((shape[0], shape[1]), dtype=np.uint8)

    # Sum the masks and convert to binary (255 for changed areas)
    change_mask = np.sum(change_masks, axis=0).astype(np.uint8)
    change_mask[change_mask > 0] = 255

    return change_mask


class Args(object):
    """docstring for ClassName"""

    def __init__(self):
        super().__init__()
        self.sam_version = "vit_h"
        self.sam_checkpoint = '/home/zsy/Programs/ChangeDetectionService/Source/Weights/DynamicEarth/weights/sam_vit_h_4b8939.pth'
        self.sam_hq_checkpoint = None
        self.use_sam_hq = True
        self.comparetor_config = {'model_type': 'DINOv2',
                                  'feature_dim': 768,
                                  'patch_size': 14}
        self.input_image_1 = None
        self.input_image_2 = None
        self.output_dir = 'outputs'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name_list = ['background', 'building']


def load_images(img1_path: str, img2_path: str) -> tuple:
    """
    Load images from the given paths.

    Args:
        img1_path (str): Path to the first image file.
        img2_path (str): Path to the second image file.

    Returns:
        tuple: Loaded images (img1, img2).
    """
    img1 = imread(img1_path)
    img2 = imread(img2_path)
    return img1, img2


def initialize_sam(use_sam_hq: bool, sam_version: str,
                   sam_checkpoint: str, sam_hq_checkpoint: str,
                   device: str):
    """
    Initialize the SAM model based on the specified parameters.

    Args:
        use_sam_hq (bool): Flag to indicate if SAM-HQ should be used.
        sam_version (str): Version of the SAM model.
        sam_checkpoint (str): Path to the SAM checkpoint file.
        sam_hq_checkpoint (str): Path to the SAM-HQ checkpoint file.
        device (str): The device to run the model on.

    Returns:
        model: Initialized SAM model.
    """
    model_registry = sam_hq_model_registry if use_sam_hq else sam_model_registry
    checkpoint = sam_hq_checkpoint if use_sam_hq else sam_checkpoint
    return model_registry[sam_version](checkpoint=checkpoint).to(device)


def setup_mask_proposal(sam) -> MaskProposal:
    """
    Set up the MaskProposal generator with hyperparameters.

    Args:
        sam: Initialized SAM model.

    Returns:
        MaskProposal: Configured MaskProposal instance.
    """
    mp = MaskProposal()
    mp.make_mask_generator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.95,
        stability_score_offset=0.9,
        box_nms_thresh=0.7,
        min_mask_region_area=0
    )
    mp.set_hyperparameters(
        match_hist=False,
        area_thresh=0.8
    )
    return mp


def detection_proc(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images
    img1, img2 = load_images(args.input_image_1, args.input_image_2)

    # Initialize SAM
    sam = initialize_sam(
        args.use_sam_hq, args.sam_version, args.sam_checkpoint,
        args.sam_hq_checkpoint, args.device
    )

    # Set up MaskProposal
    mp = setup_mask_proposal(sam)

    # Load models
    comparator_model, comparator_processor = get_model_and_processor(
        args.comparetor_config['model_type'], args.device
    )
    identifier_model, identifier_processor = get_identifier(
        'SegEarth-OV', args.device, name_list=args.name_list
    )

    # Process masks
    masks, img1_mask_num = mp.forward(img1, img2)

    # Convert RLE masks to binary numpy arrays
    masks = np.array([rle_to_mask(rle).astype(bool) for rle in masks['rles']])

    # Match masks between the two images and get class-agnostic change masks
    cmasks, img1_mask_num = bitemporal_match(img1, img2, masks, comparator_model, comparator_processor,
                                             img1_mask_num, change_confidence_threshold=145, device=args.device,
                                             model_config=args.comparetor_config)

    # Identify specific classes of change masks
    cmasks, img1_mask_num = identify(
        img1, img2, cmasks, img1_mask_num,
        identifier_model, identifier_processor,
        device=args.device
    )

    # Merge and save the final change mask
    change_mask = merge_masks(cmasks, img1.shape[:2])
    imsave(os.path.join(args.output_dir, os.path.basename(args.input_image_1)), change_mask)


def dynamic_earth_method(first_file_path, second_file_path,
                         save_file_path,
                         weights_path=None):
    args = Args()
    args.input_image_1 = first_file_path
    args.input_image_2 = second_file_path
    args.output_dir = save_file_path
    detection_proc(args)


if __name__ == "__main__":
    args = Args()
    args.input_image_1 = './demo_images/A/test_256.png'
    args.input_image_2 = './demo_images/B/test_256.png'
    detection_proc(args)
