import cv2
last_downloaded_image_name = './data/images/IMG_4.jpg'
lp_top_right_x = 204#203.546
lp_top_right_y = 334#333.68869936
lp_bottom_left_x = 337#336.742
lp_bottom_left_y = 383#382.729211087


image = cv2.imread(last_downloaded_image_name)
image = cv2.circle(image,(lp_top_right_x,lp_top_right_y), 10, (0,0,255), -1)
image = cv2.circle(image,(lp_bottom_left_x,lp_bottom_left_y), 10, (255,0,0), -1)
image = cv2.rectangle(image, (int(lp_top_right_x),int(lp_top_right_y)), (int(lp_bottom_left_x), int(lp_bottom_left_y)),(0,255,0),1)
cv2.imshow('frame', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
