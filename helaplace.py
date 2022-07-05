from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import imutils
import math
import os

# 彩色图像全局直方图均衡化
def hisEqulColor1(img):
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


# 彩色图像进行自适应直方图均衡化，代码同上的地方不再添加注释
def hisEqulColor2(img):
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	# 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input image path')
    parser.add_argument('-o', '--output', required=True, help='output image path')
    parser.add_argument('-s', default=300, type=float, help='the scale (reference value)')
    parser.add_argument('-n', default=3, type=int, help='the number of scale')
    parser.add_argument('-d', default=2, type=float, help='the dynamic, the smaller the value, the higher the contrast')
    parser.add_argument('--no_cr', action='store_true', help='do NOT do cr')


def retinex_scales_distribution(max_scale, nscales):
    scales = []
    scale_step = max_scale / nscales
    for s in range(nscales):
        scales.append(scale_step * s + 2.0)
    return scales


def CR(im_ori, im_log, alpha=128., gain=1., offset=0.):
    im_cr = im_log * gain * (
            np.log(alpha * (im_ori + 1.0)) - np.log(np.sum(im_ori, axis=2) + 3.0)[:, :, np.newaxis]) + offset
    return im_cr


def MSRCR(image_path, max_scale, nscales, dynamic=2.0, do_CR=True):
    im_ori = np.float32(cv2.imread(image_path)[:, :, (2, 1, 0)])
    scales = retinex_scales_distribution(max_scale, nscales)

    im_blur = np.zeros([len(scales), im_ori.shape[0], im_ori.shape[1], im_ori.shape[2]])
    im_mlog = np.zeros([len(scales), im_ori.shape[0], im_ori.shape[1], im_ori.shape[2]])

    for channel in range(3):
        for s, scale in enumerate(scales):
            # If sigma==0, it will be automatically calculated based on scale
            im_blur[s, :, :, channel] = cv2.GaussianBlur(im_ori[:, :, channel], (0, 0), scale)
            im_mlog[s, :, :, channel] = np.log(im_ori[:, :, channel] + 1.) - np.log(im_blur[s, :, :, channel] + 1.)

    im_retinex = np.mean(im_mlog, 0)
    if do_CR:
        im_retinex = CR(im_ori, im_retinex)

    im_rtx_mean = np.mean(im_retinex)
    im_rtx_std = np.std(im_retinex)
    im_rtx_min = im_rtx_mean - dynamic * im_rtx_std
    im_rtx_max = im_rtx_mean + dynamic * im_rtx_std

    im_rtx_range = im_rtx_max - im_rtx_min

    im_out = np.uint8(np.clip((im_retinex - im_rtx_min) / im_rtx_range * 255.0, 0, 255))

    return im_out


if __name__ == '__main__':
    #直方图
    img = cv2.imread('yt.jpg')
    img2 = img.copy()
    res2 = hisEqulColor2(img2)
    cv2.imwrite('zft.jpg', res2)

    #msrcr
    plt.close('all')
    image_path = 'yt.jpg'
    out_msrcr = MSRCR(image_path, max_scale=300, nscales=3, dynamic=2, do_CR=True)
    plt.imsave('msrcr1.jpg', out_msrcr, format='jpg')

    #对数log
    image = cv2.imread('yt.jpg')
    log_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            log_img[i, j, 0] = math.log(1 + image[i, j, 0])
            log_img[i, j, 1] = math.log(1 + image[i, j, 1])
            log_img[i, j, 2] = math.log(1 + image[i, j, 2])
    cv2.normalize(log_img, log_img, 0, 255, cv2.NORM_MINMAX)
    log_img = cv2.convertScaleAbs(log_img)
    # cv2.imshow('image', imutils.resize(image, 400))
    # cv2.imshow('log transform', imutils.resize(log_img, 400))
    #cv2.imwrite('ds.jpg',log_img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

    #Laplace
    def laplacian(imagee):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image_lap = cv2.filter2D(imagee, cv2.CV_8UC3, kernel)
        return image_lap
    img3 = cv2.imread("yt.jpg")
    dstt=laplacian(img3)
    # gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    # dst = cv2.convertScaleAbs(gray_lap)
    # cv2.imshow('laplacian', dstt)
    cv2.imwrite('lpls.jpg',dstt)

    #gama变换
    img4 = cv2.imread("yt.jpg")
    gamma=1.2 #可调
    # gamma > 1，像素强度降低，即图像变暗。
    # gamma < 1，像素强度增加，即图像变亮。
    gamma_corrected = np.array(255 * (img4 / 255) ** gamma, dtype='uint8')
    # cv2.imshow("gamma",gamma_corrected)
    #cv2.imwrite('gama1.jpg',gamma_corrected)
    #加权平均融合
    img11 = cv2.imread("zft.jpg")  # 读取图片1
    img22 = cv2.imread("lpls.jpg")  # 读取图片2

    dst1 = cv2.addWeighted(img11, 0.4, img22, 0.6,0)  # 进行加权融合处理
    cv2.imshow("0.5比0.5", dst1)
    cv2.imshow("yt",img4)
    cv2.imshow("lpls",dstt)
    cv2.imshow("zft", res2)
    cv2.imwrite('rhjq.jpg',dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
