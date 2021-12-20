import os
import cv2

img = cv2.imread(r"E:\DATASET\mao.jpg")
img = img[180:800, 100:900]
print(img.shape)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# print(img.shape)   # (1024, 1024, 3)

kuan = 50

for w in range(1, 5):
    for h in range(1, 5):
        if w * 45 - kuan < 0:
            if h * 45 - kuan < 0:
                crop_img = img[h * 150:(h + 1) * 150, w * 150:(w + 1) * 150]
                file_name = os.path.join(r'E:\DATASET\crop_image', 'crop_{}_{}.jpg'.format(w, h))
                cv2.imwrite(file_name, crop_img)
                print(file_name + 'is saved')
            else:
                crop_img = img[h * 150 - kuan:(h + 1) * 150 - kuan, w * 150:(w + 1) * 150]
                file_name = os.path.join(r'E:\DATASET\crop_image', 'crop_{}_{}.jpg'.format(w, h))
                cv2.imwrite(file_name, crop_img)
                print(file_name + 'is saved')
        else:
            if h * 45 - kuan < 0:
                crop_img = img[h * 150:(h + 1) * 150, w * 150:(w + 1) * 150]
                file_name = os.path.join(r'E:\DATASET\crop_image', 'crop_{}_{}.jpg'.format(w, h))
                cv2.imwrite(file_name, crop_img)
                print(file_name + 'is saved')
            else:
                crop_img = img[h * 150 - kuan:(h + 1) * 150 - kuan, w * 150 - kuan:(w + 1) * 150 - kuan]
                file_name = os.path.join(r'E:\DATASET\crop_image', 'crop_{}_{}.jpg'.format(w, h))
                cv2.imwrite(file_name, crop_img)
                print(file_name + 'is saved')
