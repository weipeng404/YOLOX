#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from loguru import logger
from .nilar import NilarDefectsDetection
from .nilar_casing_classes import CASING_CLASS
from .voc import AnnotationTransform

class NilarCasingDetection(NilarDefectsDetection):
	"""
	Nilar Casing Detection Dataset Object
	"""
	def __init__(
		self, 
		data_dir,
		image_set=[("Original", "train")],
		img_size=(1920, 2560),
		preproc=None,
		target_transform=AnnotationTransform(
			class_to_ind=dict(zip(CASING_CLASS, range(len(CASING_CLASS))))
		)
	):
		super(NilarCasingDetection, self).__init__(
			data_dir=data_dir,
			image_set=image_set,
			img_size=img_size,
			preproc=preproc,
			target_transform=target_transform
		)
		self._imgpath = os.path.join("%s", "Images", "%s.png")
		self._classes = CASING_CLASS

	def load_resized_img(self, index):
		img = self.load_image(index)

		# histogram equalization
		y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
		y_eq = cv2.equalizeHist(y)
		img_bgr_eq = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)
    
		r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
		resized_img = cv2.resize(
			img_bgr_eq,
			(int(img.shape[1] * r), int(img.shape[0] * r)),
			interpolation=cv2.INTER_LINEAR,
		).astype(np.uint8)

		return resized_img

	def load_image(self, index):
		img_id = self.ids[index]
		img = cv2.imread((self._imgpath % img_id).replace("casing/",""), cv2.IMREAD_COLOR)
		assert img is not None

		return img

	def evaluate_detections(self, all_boxes, output_dir=None):
		self._write_voc_results_file(all_boxes)
		IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
		mAPs = []
		for iou in IouTh:
			mAP = self._do_python_eval(output_dir, iou)
			mAPs.append(mAP)

		print("--------------------------------------------------------------")
		print("map_5095:", np.mean(mAPs))
		print("map_80:", mAPs[6])
		print("--------------------------------------------------------------")
		logger.info("mAP_80: {}, mAP_5095: {}".format(mAPs[6], np.mean(mAPs)))
		return np.mean(mAPs), mAPs[6]