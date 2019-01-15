#reads images from IMAGE_DATA_PATH, localizes the license plate, crops and equalizes the histogram and
#saves the result into PROCESSED_DATA_PATH
#for the CarsReId Dataset
import cv2
import os
import csv
import numpy as np

IMAGE_DATA_PATH = './sample_data/'
PROCESSED_DATA_PATH = './only_lp_preproc/'
pos = 1 #position in the inspection list
image = np.zeros([255,255])
sobel_k_size = 3
gauss_size = 29
clahe = cv2.createCLAHE(clipLimit = 0.8, tileGridSize = (4,4))
clahe8 = cv2.createCLAHE(clipLimit = 0.8, tileGridSize = (8,8))


def checkAspectRatio(width, height):
	error = 0.8
	if height > 10 and height < 300:
		if ((width/height) <= (4.2 + error) and (width/height) >= (4.2 - error)) or ((width/height) <= (2 + error) and (width/height) >= (2 - error)):
			return True
	return False

def onlyGtz(x):
	if x > 0:
		return x
	return 0

for img_name in os.listdir(IMAGE_DATA_PATH):
	img_name_full = os.path.join(IMAGE_DATA_PATH, img_name)
	image = cv2.imread(img_name_full)
	#grayscale conversion
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#gaussian blurring
	image_blur = cv2.GaussianBlur(image_gray,(3,gauss_size), 0)
	cv2.imshow('blurred', image_blur)
	#sobel x
	sobelx = cv2.Sobel(image_blur,cv2.CV_8U,1,0,ksize=sobel_k_size)
	cv2.imshow('sobel X', sobelx)
	#otsu thresholding
	ret, thresh = cv2.threshold(sobelx,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
	cv2.imshow('thresholded', thresh)
	#get struct. element
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))

	closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel, iterations = 2)
	opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN, kernel, iterations = 2)
	cv2.imshow('closed', closing)
	cv2.imshow('opened', opening)
	img, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	img3d = np.zeros((img.shape[0],img.shape[1],3))
	img3d[:,:,0] = img3d[:,:,1] = img3d[:,:,2] = img
	image_with_contours = cv2.drawContours(img3d, contours, -1, (0,255,0), 2)
	valid_rect_list = []
	for i in range(len(contours)):
		cnt = contours[i] 
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		width = rect[1][0]
		height = rect[1][1]

		if checkAspectRatio(width, height):
			# print rect
			valid_rect_list.append(rect)
			#prospective contours::
			# print box
		image_with_contours = cv2.drawContours(img3d,[box],0,(0,0,255),2)

	if(np.array(image).shape[0] > 0 and np.array(image).shape[1] > 0):
		cv2.imshow('contours drawn', image_with_contours)
		for j in range(len(valid_rect_list)):
			width = int(valid_rect_list[j][1][0])
			height = int(valid_rect_list[j][1][1])
			x = int(valid_rect_list[j][0][0])
			y = int(valid_rect_list[j][0][1])
			if width > 0 and height > 0:
				start_x = x - width/2
				end_x = x + width/2
				start_y = y - height/2
				end_y = y + height/2
				start_x = onlyGtz(start_x)
				start_y = onlyGtz(start_y)
				lp_proposal = image_gray[start_y:end_y, start_x:end_x]
				print 'XXXXXXXXXXXXXX',y - height/2,y + height/2,x - width/2,x + width/2
				equalized_lp_proposal = clahe.apply(lp_proposal)
				# ret, thresh = cv2.threshold(lp_proposal, 0, 255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
				cv2.imshow('frame : {}'.format(j), equalized_lp_proposal)
				print os.path.join(PROCESSED_DATA_PATH, img_name)
				cv2.imwrite(os.path.join(PROCESSED_DATA_PATH, img_name), equalized_lp_proposal)
				cv2.waitKey(0)
