import cv2
import numpy as np
import time

def cal_distance(x1, x2):
    return np.sqrt(np.sum((x1  - x2 ) ** 2))

def flat_kernel(distance, bandwidth):
    if distance <= bandwidth:
        return 1
    return 0


class MeanShiftSegment():
    def __init__(self, file_path, kernel, distance, bandwidth, threshold):
        review_data = cv2.imread(file_path)
        self.listData = np.reshape(review_data, (-1, 3)).astype(np.float64)
        self.shape = review_data.shape
        self.kernel = kernel
        self.distance = distance
        self.bandwidth = bandwidth
        self.threshold = threshold


    def shift_mode(self, xm):
        weighted_sum = 0
        total_weights = 0
        for j in range (len(self.listData)):
            kn = self.kernel(self.distance(self.listData[j],xm),self.bandwidth)
            weighted_sum = weighted_sum + self.listData[j] * kn
            total_weights = total_weights + kn
        return weighted_sum/total_weights


    def find_mode(self):
        mode = [[] for _ in range(len(self.listData))]
        self.Clusters = []
        for i in range(len(self.listData)):
            mode[i].append(self.listData[i] )
            mode[i].append(self.shift_mode(mode[i][-1]))
            
            while self.distance(mode[i][-1],mode[i][-2]) > threshold:
                mode[i].append(self.shift_mode(mode[i][-1]))
            if i % 100 == 0:
                print(f"{int((i/len(self.listData))*100)}%", end=" ==> ")
            self.Clusters.append([mode[i][-1]])
        print(f"100%\n")

    def export(self, file_out):
        shifted_img = np.reshape(self.Clusters, self.shape).astype(np.float64)
        cv2.imwrite(file_out, shifted_img)

bandwidth = 40 
threshold = .01  

start = time.time()
meanshift = MeanShiftSegment('input.jpg',flat_kernel,cal_distance,40,0.01)
meanshift.find_mode()
meanshift.export('output.jpg')
print('timer pass',time.time() - start)


