#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import pickle
from loguru import logger

from yolox.evaluators.voc_eval import voc_eval

from .nilar_classes import NILAR_CLASSES
from .voc import AnnotationTransform, VOCDetection

class NilarDefectsDetection(VOCDetection):
	"""
	Nilar Defects Detection Dataset Object

	Args:
		data_dir (string): filepath to images and annotations folder # nilar
		image_set (string): imageset to use (eg. 'train', 'val', 'test')
		img_size (tuple): (height, width) tuple used to resize the network input
		preproc (callable): preprocessing transformation to perform on the input image
		target_transform (callable): transformation to perform on the target annotation
	"""
	def __init__(
		self, 
		data_dir, 
		image_set,
		img_size = (960, 1920),
		preproc = None,
		target_transform = AnnotationTransform(
			class_to_ind = dict(zip(NILAR_CLASSES, range(len(NILAR_CLASSES))))
			),
	):
		super(VOCDetection, self).__init__(img_size)
		self.root = data_dir
		self.image_set = image_set
		self.img_size = img_size
		self.preproc = preproc
		self.target_transform = target_transform
		self._annopath = os.path.join("%s", "Annotations", "%s.xml") # eg. data_dir/Annotations/2021-10-02/OK_cropped_layer2/E1210-21395340_2.xml
		self._imgpath = os.path.join("%s", "Images", "%s.bmp") # eg. data_dir/Images/2021-10-02/OK_cropped_layer2/E1210-21395340_2.bmp
		self._classes = NILAR_CLASSES
		self.ids = list()

		for line in open(
			os.path.join(self.root, "ImageSets", "Main", self.image_set + ".txt")
		):
			self.ids.append((self.root, line.strip()))

		self.annotations = [self.load_anno_from_ids(_id) for _id in range(len(self.ids))]
		self.imgs = None

# 	def load_image(self, index):
# 		img_id = self.ids[index]
# 		img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_GRAYSCALE)
# 		assert img is not None

# 		return img

	def evaluate_detections(self, all_boxes, output_dir=None):
		self._write_voc_results_file(all_boxes)
		IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
		mAPs = []
		for iou in IouTh:
			mAP = self._do_python_eval(output_dir, iou)
			mAPs.append(mAP)

		print("--------------------------------------------------------------")
		print("map_5095:", np.mean(mAPs))
		print("map_50:", mAPs[0])
		print("--------------------------------------------------------------")
		logger.info("mAP_50: {}, \t mAP_5095: {}".format(mAPs[0], np.mean(mAPs)))
		return np.mean(mAPs), mAPs[0]

	def _get_voc_results_file_template(self):
		filename = "comp4_det_test" + "_{:s}.txt"
		filedir = os.path.join(self.root, "results", "comp4_det_test")
		if not os.path.exists(filedir):
			os.makedirs(filedir)
		filepath = os.path.join(filedir, filename)
		return filepath

	def _write_voc_results_file(self, all_boxes):
		for cls_ind, cls in enumerate(NILAR_CLASSES):
			# if cls == "NoDefects":
			# 	continue
			print("Writing {} VOC results file.".format(cls))
			filepath = self._get_voc_results_file_template().format(cls)
			with open(filepath, 'wt') as f:
				for im_ind, index in enumerate(self.ids):
					index = index[1] # image name in imageSet
					dets = all_boxes[cls_ind][im_ind]
					if dets == []:
						continue
					for k in range(dets.shape[0]):
						f.write(
							"{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
								index,
								dets[k, -1], # confidence
								dets[k, 0] + 1, # xmin
								dets[k, 1] + 1, # ymin
								dets[k, 2] + 1, # xmax
								dets[k, 3] + 1, # ymax
							)
						)

	def _do_python_eval(self, output_dir="output", iou=0.5):
		annopath = os.path.join(self.root, "Annotations", "{:s}.xml")
		imagesetfile = os.path.join(self.root, "ImageSets", "Main", self.image_set + ".txt")
		cachedir = os.path.join(self.root, "annotation_cache", self.image_set)
		if not os.path.exists(cachedir):
			os.makedirs(cachedir)

		aps = []
		print("Eval IoU: {:.2f}".format(iou))
		if output_dir is not None and not os.path.isdir(output_dir):
			os.mkdir(output_dir)
		for i, cls in enumerate(NILAR_CLASSES):
			# if cls == "NoDefects":
			# 	continue
			filepath = self._get_voc_results_file_template().format(cls)
			rec, prec, ap = voc_eval(
				detpath = filepath,
				annopath = annopath,
				imagesetfile = imagesetfile,
				classname = cls,
				cachedir = cachedir,
				ovthresh = iou,
				use_07_metric=False,
			)
			aps += [ap]
			# if iou == 0.5:
			# 	print("AP for {} = {:.4f}".format(cls, ap))
			if output_dir is not None:
				with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
					pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)

		if iou == 0.5:
			print("For IoU = 0.5, mAP = {:.4f}".format(np.mean(aps)))
			print("----------")
			print("Results:")
			for i, cls in enumerate(NILAR_CLASSES):
				print("AP for {:s}:\t{:.3f}".format(cls, aps[i]))
			print("----------")

		return np.mean(aps)