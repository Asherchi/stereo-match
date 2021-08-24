
# left_image = r'C:\Users\youqixingkong\Desktop\Middlebury\Dataset2014\Backpach_perfect\im0.png'
# right_imgae = r'C:\Users\youqixingkong\Desktop\Middlebury\Dataset2014\Backpach_perfect\im1.png'
# Limage = cv2.imread(left_image,0)
# Rimage = cv2.imread(right_imgae,0)
#
# sift = cv2.SIFT_create()
import cv2 as cv

left_image = r'C:\Users\youqixingkong\Desktop\Middlebury\Dataset2014\Backpach_perfect\im0.png'
right_image = r'C:\Users\youqixingkong\Desktop\Middlebury\Dataset2014\Backpach_perfect\im1.png'

# 1.加载图片
left_image = cv.imread(left_image)
left_image = cv.resize(left_image, (640, 640))
right_image = cv.imread(right_image)
right_image = cv.resize(right_image, (640, 640))
cv.imshow('left_image', left_image)
cv.imshow('right_image', right_image)
# 2.灰度化
left_gray = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
cv.imshow('left_Gray', left_gray)
right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
cv.imshow('right_Gray', right_gray)
# 3.特征点检测
sift = cv.SIFT_create()
# left_keypoints = sift.detect(left_gray)
# right_keypoints = sift.detect(right_gray)
left_keypoints, left_descriptor = sift.detectAndCompute(left_gray,None)
right_keypoints, right_descriptor = sift.detectAndCompute(right_gray, None)
# 4.绘制特征点
left_mixture = left_image.copy()
cv.drawKeypoints(left_image, left_keypoints, left_mixture,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
right_mixture = right_image.copy()
cv.drawKeypoints(right_image, right_keypoints, right_mixture,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('left_Mixture', left_mixture)
cv.imshow("right_mixture",right_mixture)
cv.waitKey()

Matcher = cv.BFMatcher()
# DMatch数据类型是俩个与原图像特征点最接近的俩个特征点（match返回的是最匹配的）只有这俩个特征点的欧式距离小于一定值的时候才会认为匹配成功
# queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
# trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
# distance：代表这一对匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
raw_matches = Matcher.knnMatch(left_descriptor, right_descriptor, k=2)
good_matches = []
for m1, m2 in raw_matches:
    #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < 0.65 * m2.distance:
        good_matches.append([m1])
# end = time.time()
# print("匹配点匹配运行时间:%.2f秒" % (end - start))

matches = cv.drawMatchesKnn(left_image, left_keypoints, right_image, right_keypoints, good_matches, None, flags=2)

cv.imshow("matches", matches)
cv.waitKey()
# plt.figure()
# plt.imshow(matches)


