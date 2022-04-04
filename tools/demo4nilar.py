import argparse
import os
import time
from loguru import logger

import cv2
from lxml import etree

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import NILAR_CLASSES
from yolox.exp import get_exp
from yolox.tools.demo import Predictor, get_image_list, image_demo
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
    parser.add_argument("--save_xml", action="store_true", help="whether to save the inference results to xml files")
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

def image_demo(predictor, vis_folder, xml_folder, path, current_time, save_result, save_xml):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()

    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

        if save_xml:
            write_xml(
                xml_folder=xml_folder, 
                image_name=image_name, 
                img_info=img_info, 
                output=outputs[0], 
                cls_conf=predictor.confthre,
                class_names=NILAR_CLASSES,
            )


def write_xml(xml_folder, image_name, img_info, output, cls_conf, class_names):
    root = etree.Element("annotation")
    etree.SubElement(root, "folder").text=xml_folder
    etree.SubElement(root, "filename").text=os.path.basename(image_name)
    etree.SubElement(root, "path").text=os.path.abspath(image_name)
    etree.SubElement(etree.SubElement(root, "source"), "database").text="Unknown"

    size = etree.Element("size")
    etree.SubElement(size, "width").text=str(img_info["width"])
    etree.SubElement(size, "height").text=str(img_info["height"])
    etree.SubElement(size, "depth").text="1"
    root.append(size)

    etree.SubElement(root, "segmented").text="0"

    if output is not None:
        output = output.cpu()

        bboxes = output[:, 0:4]
        bboxes /= img_info["ratio"] # resize
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(cls[i])
            score = scores[i]
            if score < cls_conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            class_name = class_names[cls_id]

            object = etree.Element("object")
            etree.SubElement(object, "name").text=class_name
            etree.SubElement(object, "pose").text="Unspecified"
            etree.SubElement(object, "truncated").text="0"
            etree.SubElement(object, "difficult").text="0"
            bndbox = etree.Element("bndbox")
            etree.SubElement(bndbox, "xmin").text=str(x0)
            etree.SubElement(bndbox, "ymin").text=str(y0)
            etree.SubElement(bndbox, "xmax").text=str(x1)
            etree.SubElement(bndbox, "ymax").text=str(y1)
            object.append(bndbox)
            root.append(object)

    tree = etree.ElementTree(root)
    xml_filename = os.path.join(xml_folder, os.path.basename(image_name)[:-4] + '.xml')
    logger.info("Saving predicted label in xml format in {}".format(xml_filename))
    tree.write(xml_filename, pretty_print=True)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    xml_folder = None
    if args.save_xml:
        img_path = os.path.normpath(args.path)
        img_path_list = img_path.split(os.sep)
        img_path_list[img_path_list.index('Images')] = "Annotations"
        if img_path.endswith('.bmp'):
            xml_folder = '/'.join(img_path_list[:-1]) + "_pred"
        else:
            xml_folder = '/'.join(img_path_list) + "_pred"
        os.makedirs(xml_folder, exist_ok=True)

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
                predictor=predictor, 
                vis_folder=vis_folder, 
                xml_folder=xml_folder,
                path=args.path, 
                current_time=current_time, 
                save_result=args.save_result,
                save_xml=args.save_xml,
            )
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)
