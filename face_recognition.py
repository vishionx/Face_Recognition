from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
import cv2
import numpy
import torch
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
#cv2.namedWindow("img")
##cv2.imshow("img",img)
cv2.waitKey(0)
cr_img = img[bounding_box[1]: bounding_box[1]+bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2]]
print(cr_img.shape)
cv2.imwrite("face_crop2.jpg", cr_img)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
exp_img = numpy.expand_dims(cr_img, axis=0)
exp_img = exp_img.transpose((0, 3, 1, 2))  # Change the order of dimensions to [batch_size, channels, height, width]
tensor = torch.from_numpy(exp_img.astype(numpy.float32))
img_embedding = resnet(tensor)