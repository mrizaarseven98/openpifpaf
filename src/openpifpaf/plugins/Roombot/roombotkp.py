
import argparse
import torch
import numpy as np
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from openpifpaf.datasets import DataModule
from openpifpaf import encoder, headmeta, metric, transforms
from openpifpaf.datasets import collate_images_anns_meta, collate_images_targets_meta
from openpifpaf.plugins.coco import CocoDataset as CocoLoader

from .constants import get_constants
from .metrics import MeanPixelError


class RoombotKp(DataModule):
    train_annotations = '/home/riza/Desktop/data_Roombot/annotations/train/train.json'
    val_annotations = '/home/riza/Desktop/data_Roombot/annotations/test/test.json'
    #val_annotations=train_annotations
    eval_annotations = train_annotations
    train_image_dir = '/home/riza/Desktop/data_Roombot/images/train/train/'
    val_image_dir = '/home/riza/Desktop/data_Roombot/images/test/'
    #val_image_dir=train_image_dir
    eval_image_dir = train_image_dir
    """
Interface for custom data.

This module handles datasets and is the class that you need to inherit from for your custom dataset.
This class gives you all the handles so that you can train with a new –dataset=mydataset.
The particular configuration of keypoints and skeleton is specified in the headmeta instances
"""


    n_images = None
    square_edge = 500
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel

    eval_annotation_filter = True
    eval_long_edge = 0  # set to zero to deactivate rescaling
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()
        if self.weights is not None:
            caf_weights = []
            for bone in self.ROBOT_SKELETON:
                caf_weights.append(max(self.weights[bone[0] - 1],
                                       self.weights[bone[1] - 1]))
            w_np = np.array(caf_weights)
            caf_weights = list(w_np / np.sum(w_np) * len(caf_weights))
        else:
            caf_weights = None
        cif = headmeta.Cif('cif', 'Roombot',
                           keypoints=self.ROBOT_KEYPOINTS,
                           sigmas=self.ROBOT_SIGMAS,
                           pose=self.ROBOT_POSE,
                           draw_skeleton=self.ROBOT_SKELETON,
                           score_weights=self.ROBOT_SCORE_WEIGHTS,
                           training_weights=self.weights)
        caf = headmeta.Caf('caf', 'Roombot',
                           keypoints=self.ROBOT_KEYPOINTS,
                           sigmas=self.ROBOT_SIGMAS,
                           pose=self.ROBOT_POSE,
                           skeleton=self.ROBOT_SKELETON,
                           training_weights=caf_weights)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Roombot')

        group.add_argument('--Roombot-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--Roombot-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--Roombot-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--Roombot-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--Roombot-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--Roombot-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--Roombot-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--Roombot-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--Roombot-no-augmentation',
                           dest='Roombot_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--Roombot-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--Roombot-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--Roombot-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--Roombot-bmin',
                           default=cls.b_min, type=int,
                           help='b minimum in pixels')
        group.add_argument('--Roombot-apply-local-centrality-weights',
                           dest='Roombot_apply_local_centrality',
                           default=False, action='store_true',
                           help='Weigh the CIF and CAF head during training.')

        # evaluation
        assert cls.eval_annotation_filter
        group.add_argument('--Roombot-no-eval-annotation-filter',
                           dest='Roombot_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--Roombot-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--Roombot-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--Roombot-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)
        group.add_argument('--Roombot-use-10-kps', default=False, action='store_true',
                           help=(' '))

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # Roombot specific
        cls.train_annotations = args.Roombot_train_annotations
        cls.val_annotations = args.Roombot_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.Roombot_train_image_dir
        cls.val_image_dir = args.Roombot_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.Roombot_square_edge
        cls.extended_scale = args.Roombot_extended_scale
        cls.orientation_invariant = args.Roombot_orientation_invariant
        cls.blur = args.Roombot_blur
        cls.augmentation = args.Roombot_augmentation  # loaded by the dest name
        cls.rescale_images = args.Roombot_rescale_images
        cls.upsample_stride = args.Roombot_upsample
        cls.min_kp_anns = args.Roombot_min_kp_anns
        cls.b_min = args.Roombot_bmin
        if True:#args.Roombot_use_10_kps:
            (cls.ROBOT_KEYPOINTS, cls.ROBOT_SKELETON, cls.HFLIP, cls.ROBOT_SIGMAS, cls.ROBOT_POSE,
             cls.ROBOT_CATEGORIES, cls.ROBOT_SCORE_WEIGHTS) = get_constants()
        # evaluation
        cls.eval_annotation_filter = args.Roombot_eval_annotation_filter
        cls.eval_long_edge = args.Roombot_eval_long_edge
        cls.eval_orientation_invariant = args.Roombot_eval_orientation_invariant
        cls.eval_extended_scale = args.Roombot_eval_extended_scale
        
        if args.Roombot_apply_local_centrality:
            '''
            if args.Roombot_use_10_kps:
                raise Exception("Applying local centrality weights only works with 66 kps.")
            cls.weights = training_weights_local_centrality
            '''

        else:
            cls.weights = None

    def _preprocess(self):
        encoders = (encoder.Cif(self.head_metas[0], bmin=self.b_min),
                    encoder.Caf(self.head_metas[1], bmin=self.b_min))

        if not self.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(self.square_edge),
                transforms.CenterPad(self.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.2 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.33 * self.rescale_images,
                             1.33 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            # transforms.AnnotationJitter(),
            transforms.RandomApply(transforms.HFlip(self.ROBOT_KEYPOINTS, self.HFLIP), 0.5),
            rescale_t,
            transforms.RandomApply(transforms.Blur(), self.blur),
            transforms.RandomChoice(
                [transforms.RotateBy90(),
                 transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.2],
            ),
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.CenterPad(self.square_edge),
            transforms.MinSize(min_side=32.0),
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders(encoders),
        ])


    def train_loader(self):
        train_data = CocoLoader(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoLoader(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                transforms.DeterministicEqualChoice([
                    transforms.RescaleAbsolute(cls.eval_long_edge),
                    transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = transforms.DeterministicEqualChoice([
                None,
                transforms.RotateBy90(fixed_angle=90),
                transforms.RotateBy90(fixed_angle=180),
                transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return transforms.Compose([
            *self.common_eval_preprocess(),
            transforms.ToAnnotations([
                transforms.ToKpAnnotations(
                    self.ROBOT_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                transforms.ToCrowdAnnotations(self.ROBOT_CATEGORIES),
            ]),
            transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoLoader(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=collate_images_anns_meta)

# TODO: make sure that 24kp flag is activated when evaluating a 24kp model
    def metrics(self):
        return [metric.Coco(
            COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
            keypoint_oks_sigmas=self.ROBOT_SIGMAS
        ), MeanPixelError()]
