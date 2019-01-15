#reads localized and equalized license plate images, then finds contours and segments digits on the plate
#saves two copies, a 32x32 random gray-value padded grayscale digit image (for evaluating inference using SVHN)
#and a 28x28 zero-padded binarized digit image for inference using MNIST

import cv2
import os
import csv
import numpy as np

# IMAGE_DATA_PATH = './sample_data/'
IMAGE_DATA_PATH = './only_lp_preproc/'
GRAY_SAVE_PATH = './gray_digits/'
BINARY_SAVE_PATH = './binary_digits/'
pos = 1 #position in the inspection list
image = np.zeros([255,255])
sobel_k_size = 3
gauss_size = 29
clahe = cv2.createCLAHE(clipLimit = 0.8, tileGridSize = (4,4))
clahe8 = cv2.createCLAHE(clipLimit = 0.8, tileGridSize = (8,8))

locality = 3 #number of pixels around the contour to be selected in the box
def checkAspectRatio(width, height):
	error = 0.25
	aspect_ratio = (width / (height + 1))
	if (aspect_ratio > 0.25) and aspect_ratio < 0.75 and width < 20 and height < 28 and width > 3 and height > 10:
		return True
	# if height > 10 and height < 300:
	# 	if ((width/height) <= (4.2 + error) and (width/height) >= (4.2 - error)) or ((width/height) <= (2 + error) and (width/height) >= (2 - error)):
	# 		return True
	return False

def onlyGtz(x):
	if x > 0:
		return x
	return 0

for img_name in os.listdir(IMAGE_DATA_PATH):
	img_name_full = os.path.join(IMAGE_DATA_PATH, img_name)
	image = cv2.imread(img_name_full)
	cv2.imshow('image', image)
	#grayscale conversion
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#gaussian blurring
	# image_blur = cv2.GaussianBlur(image_gray,(3,gauss_size), 0)
	# cv2.imshow('blurred', image_blur)
	#sobel x
	# sobelx = cv2.Sobel(image_blur,cv2.CV_8U,1,0,ksize=sobel_k_size)
	# cv2.imshow('sobel X', sobelx)
	#otsu thresholding
	ret, thresh = cv2.threshold(image_gray,115,255,cv2.THRESH_BINARY_INV)
	cv2.imshow('thresholded', thresh)
	print 'thresh.shape', thresh.shape
	#get struct. element

	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	# thresh = cv2.dilate(thresh, kernel, iterations= 1)
	# thresh= cv2.erode(thresh, kernel, iterations = 1)
	# cv2.imshow('eroded', thresh)
	
	# cv2.waitKey(0)
	# exit()

	img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	img3d = np.zeros((img.shape[0],img.shape[1],3))
	img3d[:,:,0] = img3d[:,:,1] = img3d[:,:,2] = img
	image_with_contours = cv2.drawContours(img3d, contours, -1, (0,255,0), 1)
	valid_rect_list = []
	print len(contours)
	count = 0
	for i in range(len(contours)):
		cnt = contours[i] 
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		width = rect[1][0]
		height = rect[1][1]
		# print width/(height+1), width * height
		if checkAspectRatio(width, height) and count < 10:
			valid_rect_list.append(rect)
			print width, height
			#prospective contours::
			image_with_contours = cv2.drawContours(img3d,[box],0,(0,0,255),1)
			count += 1
			cv2.imshow('contours', image_with_contours)
			cv2.waitKey(0)
	if(np.array(image).shape[0] > 0 and np.array(image).shape[1] > 0):
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
				temp_locality = locality
				if start_y - locality < 0:
					temp_locality = start_y
				if start_x - locality < 0:
					temp_locality = start_x
				lp_proposal = image[start_y - temp_locality:end_y + temp_locality, start_x - temp_locality:end_x + temp_locality]
				lp_proposal_binary = thresh[start_y - temp_locality:end_y + temp_locality, start_x - temp_locality:end_x + temp_locality]
				print 'XXXXXXXXXXXXXX',y - height/2,y + height/2,x - width/2,x + width/2
				# equalized_lp_proposal = clahe.apply(lp_proposal)
				# ret, thresh = cv2.threshold(lp_proposal, 0, 255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
				cv2.imshow('frame : {}'.format(j), lp_proposal)
				cv2.imshow('frame2 : {}'.format(j), lp_proposal_binary)
				#creating a 32x32 image from the detected digit::
				#using vstack and hstack::
				## TODO: seed a random number for the background color
				random_number = np.random.randint(low = 100, high = 256)
				test_image = np.vstack([random_number * np.ones((int((32 - lp_proposal.shape[0]) / 2),lp_proposal.shape[1],3)), lp_proposal])
				test_image = np.vstack([test_image, random_number * np.ones((int((32 - lp_proposal.shape[0]) / 2),lp_proposal.shape[1],3))])
				test_image = np.hstack([test_image, random_number * np.ones((32, int((32 - lp_proposal.shape[1]) / 2),3))])
				test_image = np.hstack([random_number * np.ones((32, int((32 - lp_proposal.shape[1]) / 2),3)),test_image])	
				
				test_image_binary = np.vstack([np.zeros((int((28 - lp_proposal.shape[0]) / 2),lp_proposal.shape[1])), lp_proposal_binary])
				test_image_binary = np.vstack([test_image_binary, np.zeros((int((28 - lp_proposal.shape[0]) / 2),lp_proposal.shape[1]))])
				test_image_binary = np.hstack([test_image_binary, np.zeros((28, int((28 - lp_proposal.shape[1]) / 2)))])
				test_image_binary = np.hstack([np.zeros((28, int((28 - lp_proposal.shape[1]) / 2))),test_image_binary])	
				print lp_proposal.shape
				print test_image.shape
				cv2.imwrite(os.path.join(GRAY_SAVE_PATH, 'test_gray{}.png'.format(pos)),test_image)
				cv2.imshow('testImg_{}'.format(pos), test_image)

				cv2.imwrite(os.path.join(BINARY_SAVE_PATH,'test_binary{}.png'.format(pos)),test_image_binary)
				cv2.imshow('testImgBin_{}'.format(pos), test_image)
				pos += 1
				# for i in range(0,test_image.shape[0]):
					# print test_image[i]
				cv2.waitKey(0)
				cv2.destroyAllWindows()
