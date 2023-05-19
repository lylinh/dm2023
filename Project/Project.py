import cv2
import numpy as np
import math
import time

start = time.time()


def cal_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (distance / bandwidth)**2)

def flat_kernel(distance, bandwidth):
    if distance <= bandwidth:
        return 1
    return 0

bandwidth = 50 
threshold = 15

review_data = cv2.imread('input_image.jpg')
listdata = np.reshape(review_data, (-1, 3)).astype(np.float64)
print(listdata)

def shiftMode(listdata, nor, xm):
    weighted_sum = 0
    total_weights = 0
    for j in range (len(listdata)):
        kernel = flat_kernel(cal_distance(listdata[j],xm),bandwidth)
        weighted_sum = weighted_sum + listdata[j]*kernel
        total_weights = total_weights + kernel
    return weighted_sum/total_weights



#define mode
mode = [[] for _ in range(len(listdata))]
C = []
for i in range(len(listdata)):
    mode[i].append(listdata[i])
    mode[i].append(shiftMode(listdata,listdata[i],mode[i][-1]))

    while (cal_distance(mode[i][-1],mode[i][-2]) > threshold):
        mode[i].append(shiftMode(listdata,listdata[i],mode[i][-1]))

    C.append([mode[i][-1]])

print(C)

shifted_img = np.reshape(C, review_data.shape).astype(np.float64)
# Save the resulting image
cv2.imwrite('out_image.jpg', shifted_img)

end = time.time()
print('timer pass',end - start)