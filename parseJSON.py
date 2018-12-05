import json
import os
import urllib
import csv

FILE_NAME = './data/Indian_Number_plates.json'
IMAGE_PATH_NAME = './data/images/'
ANNOT_PATH_NAME = './data/annotations/bb.csv'

annot_file = open(ANNOT_PATH_NAME, 'w')

#read the number of existing images in the directory!
n_images = len(os.listdir(IMAGE_PATH_NAME))
print n_images

# def download_img(url,fileName, save_path_name):
#     # download the image from the specified url
#     image = urllib.URLopener()
#     image.retrieve(url,comicName)  # download comicName at URL


def download_img(url, save_path_name):
    # download the image from the specified url
    image = urllib.urlopen(url).read()
    with open(save_path_name, 'wb') as f:
    	f.write(image)


with open(FILE_NAME) as f:
	raw_data = f.read()

error_count = 0
successive_error_count = 0
i = 0
downloaded_flag = True
last_downloaded_image_name = ""


writer = csv.writer(annot_file, delimiter=',')

for line in raw_data.split('\n'):
	i += 1
	if i > n_images:
		json_line = json.loads(line)
		img_url = json_line['content']
		try:
			n_images += 1
			image_path = IMAGE_PATH_NAME + 'IMG_'+ str(n_images) + '.jpg'
			download_img(img_url, image_path)
			print 'downloaded!	' + img_url
			downloaded_flag = True
			successive_error_count = 0
			last_downloaded_image_name = image_path
		except IOError:
			error_count += 1
			downloaded_flag = False
			print "Error 404, Image not Found!"
			if successive_error_count > 3:
				print "Something's Wrong I Can Feel It!"
				exit()
		if downloaded_flag:
			img_annot = json_line['annotation'][0]
			lp_top_right_x = img_annot['points'][0]['x'] * img_annot["imageHeight"]
			lp_top_right_y = img_annot['points'][0]['y'] * img_annot["imageWidth"]

			lp_bottom_left_x = img_annot['points'][1]['x'] * img_annot["imageHeight"]
			lp_bottom_left_y = img_annot['points'][1]['y'] * img_annot["imageWidth"]
			lp_label = img_annot['label'][0]
			image_and_bb = last_downloaded_image_name + ',' + str(lp_top_right_x) + ',' + str(lp_top_right_y) + ',' + str(lp_bottom_left_x) + ',' + str(lp_bottom_left_y) + ',' + lp_label
			imgBBList = image_and_bb.split(',')

			print "last_downloaded_image_name = {}, lp_top_right_x = {}, lp_top_right_y = {}, lp_bottom_left_x = {}, lp_bottom_left_y = {}".format(last_downloaded_image_name, lp_top_right_x, lp_top_right_y, lp_bottom_left_x, lp_bottom_left_y)
			# annot_file.write(image_and_bb)
			writer.writerow(imgBBList)
# '''{"content": "http://com.dataturks.a96-i23.open.s3.amazonaws.com/2c9fafb0646e9cf9016473f1a561002a/77d1f81a-bee6-487c-aff2-0efa31a9925c____bd7f7862-d727-11e7-ad30-e18a56154311.jpg.jpeg",
# "annotation":[{"label":["number_plate"],"notes":"","points":[{"x":0.7220843672456576,"y":0.5879828326180258},{"x":0.8684863523573201,"y":0.6888412017167382}],"imageWidth":806,"imageHeight":466}],"extras":null}
# '''