#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
# import xml.etree.ElementTree as ET

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
		img_size,
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
			os.path.join(self.root, "ImageSets", "Main", image_set + ".txt")
		):
			self.ids.append((self.root, line.strip()))

		self.annotations = [self.load_anno_from_ids(_id) for _id in range(len(self.ids))]
		self.imgs = None

	def load_image(self, index):
		img_id = self.ids[index]
		img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_GRAYSCALE)
		assert img is not None

		return img

