#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import time
from loguru import logger
import json

import cv2
from lxml import etree

import torch

import yolox
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import NILAR_CLASSES, CASING_CLASS
from yolox.exp import get_exp
from yolox.tools.demo import Predictor, image_demo
from yolox.utils import get_model_info, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser(description="YOLOX Demo for Nilar!")
    parser.add_argument("demo", default="image", help="demo input type")
    parser.add_argument("-expn", "--experiment-name", default=None, type=str)
    parser.add_argument("-f", "--exp_file", default="exps/example/yolox_voc/yolox_voc_s.py", type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", type=str, help="checkpoint model for evaluation")
    parser.add_argument("--path", help="path to images")
    parser.add_argument("--layer", default=None, type=int, nargs='*', help="predict only specific layers")
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

def get_image_list(path, layer=None):
    image_list = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            image_path = os.path.join(maindir, filename)
            image_name, ext = os.path.splitext(image_path)
            if ext in IMAGE_EXT:
                if layer is None:
                    image_list.append(image_path)
                elif len(layer) < 3:
                    image_layer = int(image_name.split('_')[-1])
                    layer_range = [i for i in range(layer[0], layer[-1]+1)]
                    if image_layer in layer_range:
                        image_list.append(image_path)
    return image_list

def postprocess_output(output, img_info):
    res = {}
    ratio = img_info["ratio"]
    output = output.cpu().numpy()

    res['bboxes'] = output[:, 0:4] / ratio
    res['classes'] = output[:, 6]
    res['scores'] = output[:, 4] * output[:, 5] 
    return res

def image_demo(predictor, report_folder, vis_folder, xml_folder, path, layer, current_time, save_result, save_xml):
    if os.path.isdir(path):
        files = get_image_list(path, layer)
    else:
        files = [path]
    files.sort()

    report_save_dir = os.path.join(report_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
    os.makedirs(report_save_dir, exist_ok=True)
    report_path = os.path.join(report_save_dir, "InferenceReport.json")
    result = {}
    max_of_all_scores = 0

    for image_name in files:
        outputs, img_info = predictor.inference(image_name)

        image_id = os.path.basename(image_name)
        result[image_id] = {'bboxes': []}
        result[image_id]['layer_score'] = 0

        if outputs[0] is None:
            result_image = img_info['raw_img']
        else:
            boxes = postprocess_output(outputs[0], img_info)
            
            result[image_id]['bboxes'] = process_boxes(boxes, predictor.confthre, predictor.cls_names)
            max_score = max(boxes['scores'])
            max_of_all_scores = max(max_of_all_scores, max_score)
            if max_score >= predictor.confthre:
                result[image_id]['layer_score'] = str(max_score)

            result_image = vis(
                img=img_info['raw_img'], 
                boxes=boxes['bboxes'], 
                scores=boxes['scores'], 
                cls_ids=boxes['classes'],
                conf=predictor.confthre,
                class_names=predictor.cls_names,
            )

        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

        if save_xml:
            write_xml(
                xml_folder=xml_folder, 
                image_name=image_name, 
                img_info=img_info, 
                boxes = None if outputs[0] is None else postprocess_output(outputs[0], img_info),
                cls_conf=predictor.confthre,
                class_names=predictor.cls_names,
            )
    result['defect_score_of_folder'] = str(max_of_all_scores)
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)

def process_boxes(boxes, cls_conf, class_names):
    output_boxes = []
    for i, box in enumerate(boxes['bboxes']):
        cls_id = int(boxes['classes'][i])
        class_name = class_names[cls_id]

        score = boxes['scores'][i]
        if score < cls_conf:
            continue

        output_boxes.append({'class': class_name, 'score': str(score), 'bbox': box.astype(int).tolist()})
    return output_boxes


def write_xml(xml_folder, image_name, img_info, boxes, cls_conf, class_names):
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

    if boxes is not None:
        for i, box in enumerate(boxes['bboxes']):
            cls_id = int(boxes['classes'][i])
            class_name = class_names[cls_id]

            score = boxes['scores'][i]
            if score < cls_conf:
                continue

            x0 = round(box[0])
            y0 = round(box[1])
            x1 = round(box[2])
            y1 = round(box[3])

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

    report_folder = os.path.join(file_name, "report")
    os.makedirs(report_folder, exist_ok=True)

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

        cls_names = NILAR_CLASSES
        if args.demo == "image_original":
            cls_names = CASING_CLASS
        predictor = Predictor(
            model=model, exp=exp, cls_names=cls_names, trt_file=None, decoder=None,
            device=args.device, fp16=False, legacy=False,
        )

        image_demo(
                predictor=predictor, 
                report_folder=report_folder,
                vis_folder=vis_folder, 
                xml_folder=xml_folder,
                path=args.path, 
                layer=args.layer,
                current_time=time.localtime(), 
                save_result=args.save_result,
                save_xml=args.save_xml,
            )

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)
