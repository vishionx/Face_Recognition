from mtcnn import MTCNN
import cv2
img = cv2.imread("per2.jpg")
detector = MTCNN()
result = detector.detect_faces(img)
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']
cv2.rectangle(img,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)
cv2.imwrite("face_drawn2.jpg", img)
cv2.namedWindow("img")
cv2.imshow("img",img)
cv2.waitKey(0)
cr_img = img[bounding_box[1]: bounding_box[1]+bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2]]
cv2.imwrite("face_crop2.jpg", cr_img)
