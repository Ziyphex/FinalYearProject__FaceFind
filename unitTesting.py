"""
Title: Unit Testing
Author: Conor Cohen Farrell
Last Edited: 27 March 2019
GitHub: https://github.com/Ziyphex
Description: Unit Testing for Face Find App
"""

import unittest
from face_find_app import UiMainWindow
import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
landmark_model = "trained_models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(landmark_model)

class TestSystem(unittest.TestCase):

	def test_get_candidate_id(self):
		nextCanID = UiMainWindow.get_candidate_id()
		self.assertEqual(nextCanID, 4)

	def test_rect_to_bb(self):
		img = cv2.imread("will_smith.jpg", 1)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			(bX, bY, bW, bH) = UiMainWindow.rect_to_bb(rect)
			self.assertEqual((bX, bY, bW, bH), (701, 285, 535, 535))

	def test_get_euler_angle(self):
		img = cv2.imread("will_smith.jpg", 1)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			shape = predictor(gray, rect)
			shape = UiMainWindow.make_np_from_shape(shape)
			result = UiMainWindow.get_euler_angle(shape, img)
			#print(result[2][0])
			self.assertEqual((result[0])[0], -5.5380575709482285)
			self.assertEqual((result[1])[0], -19.034219586402706)
			self.assertEqual((result[2])[0], -6.737566633012812)

	def test_make_np_from_shape(self):
		img = cv2.imread("will_smith.jpg", 1)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			shape = predictor(gray, rect)
			shape = UiMainWindow.make_np_from_shape(shape)
			#print(shape[0])
			self.assertEqual(shape[0][0], 635)
			self.assertEqual(shape[0][1], 513)

			#print(shape[7])
			self.assertEqual(shape[7][0], 974)
			self.assertEqual(shape[7][1], 873)

			#print(shape[15])
			self.assertEqual(shape[15][0], 1155)
			self.assertEqual(shape[15][1], 536)

			#print(shape[21])
			self.assertEqual(shape[21][0], 945)
			self.assertEqual(shape[21][1], 392)

			#print(shape[39])
			self.assertEqual(shape[39][0], 923)
			self.assertEqual(shape[39][1], 482)

if __name__ == '__main__':
	unittest.main()
