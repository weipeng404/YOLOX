import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import NILAR_CLASSES
from yolox.exp import get_exp
from yolox.tools.demo import Predictor, image_demo
from yolox.utils import get_model_info


def make_parser():
    parser = argparse.ArgumentParser(description="YOLOX Demo for Nilar!")
    parser.add_argument("demo", default="image", help="demo type")
    parser.add_argument("-expn", "--experiment-name", default=None, type=str)
    parser.add_argument("-f", "--exp_file", default="exps/example/yolox_voc/yolox_voc_s.py", type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/yolox_voc_s/best_ckpt.pth", type=str, help="checkpoint model for evaluation")
    parser.add_argument("--path", help="path to images")
    parser.add_argument("--conf", default=0.3, type=float, help="test confidence")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=[960, 1920], type=int, nargs='*', help="test height and width")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference results")
    parser.add_argument("--device", default="cpu", type=str, help="device to run the model, [cpu/gpu]")

    # the following arguments will not be used in most cases, added to be compatible with the class Predictor
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    if len(args.tsize) == 1:
        exp.test_size = (args.tsize[0], args.tsize[0])
    elif len(args.tsize) == 2:
        exp.test_size = (args.tsize[0], args.tsize[1])

    # model init
    model = exp.get_model() 
    logger.info("Model summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval() # evaluation mode

    # load evaluation model
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        logger.info("loaded checkpoint file from {}".format(args.ckpt))
        model.load_state_dict(ckpt["model"])

        predictor = Predictor(
            model=model, exp=exp, cls_names=NILAR_CLASSES, trt_file=None, decoder=None,
            device=args.device, fp16=False, legacy=False,
        )

        current_time = time.localtime()
        if args.demo == "image":
            image_demo(
                predictor=predictor, vis_folder=vis_folder, path=args.path, current_time=current_time, save_result=args.save_result,
            )
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)
