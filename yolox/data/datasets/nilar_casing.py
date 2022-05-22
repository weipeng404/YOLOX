#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from .nilar import NilarDefectsDetection
from .voc import AnnotationTransform

CASING_CLASS = ("casing",)

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