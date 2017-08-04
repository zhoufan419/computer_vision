import cv2
import numpy as np
import matplotlib.pyplot as plt
#from skimage import exposure


#Input images
image1 = cv2.imread('input\lena.png')
image2 = cv2.imread('input\leaf.png')
cv2.imwrite('output\ps1-1-a-1.png',image1)
cv2.imwrite('output\ps1-1-a-2.png',image2)


#Color planes
##Swap the red pixels and blue pixels of image1
b,g,r = cv2.split(image1)
image1_swap = cv2.merge([r,g,b])
cv2.imwrite('output\ps1-2-a-1.png',image1_swap)


##Create a monochrome img1_green by selecting the green channel of image1
img1_green = image1.copy()[:,:,1]
cv2.imwrite('output\ps1-2-b-1.png',img1_green)


##Create a monochrome img1_red by selecting the red channel of image1
img1_red = image1.copy()[:,:,2]
cv2.imwrite('output\ps1-2-c-1.png',img1_red)


##Grayscale conversion of the original image1
gray = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)#0.114*B+0.587*G+0.299*R
cv2.imwrite('temp\image1_BGR2GRAY.png',gray)


#Replacement of pixels
##Take center square region of 100x100 pixels of ps1-2-b-1.png and insert it into a monochome version of image2
img2_red = image2.copy()[:,:,2]
img2_red[image2.shape[0]//2-50:image2.shape[0]//2+51, image2.shape[1]//2-50:image2.shape[1]//2+51] =img1_green[image1.shape[0]//2-50:image1.shape[0]//2+51, image1.shape[1]//2-50:image1.shape[1]//2+51]
cv2.imwrite('output\ps1-3-a-1.png',img2_red)


#Arithmetic and Geometric operations
##The min and max of the pixel values of img1_green, the mean? the standard deviation
standard_deviation = np.round(np.std(img1_green))
mean_value = np.round(np.mean(img1_green))
print ('The standard deviation of img1_green is', standard_deviation)
print ('The mean of img1_green is', mean_value)


##Subtract the mean from all pixels, then divide by standard deviation, then multiply by 10
img_manipulation = img1_green.copy()
subtract_mean = img_manipulation-np.ones(img_manipulation.shape, dtype=np.int8)*mean_value
divide_std_x10 = np.round(subtract_mean/standard_deviation*10)
plus_mean = divide_std_x10+np.ones(img_manipulation.shape, dtype=np.int8)*mean_value
###Here all plus mean value are in [0,255), no need to rescale
img_manipulation = plus_mean
cv2.imwrite('output\ps1-4-b-1.png',img_manipulation)


##Shift img1_green to the left by 2 pixels
img_shift = img1_green.copy()
rows,cols = img_shift.shape
M = np.float32([[1,0,-2],[0,1,0]])
img_shift = cv2.warpAffine(img_shift,M,(cols,rows))
cv2.imwrite('output\ps1-4-c-1.png',img_shift)


##Subtract img_shift of img1_green from the original img1_green
###Since some pixels are negative after difference, we need to expand data range to int16 to stroe negative values.
difference = np.int16(img_shift) - np.int16(img1_green)
cv2.imwrite('temp\ps1-4-d-1_without_scaling.png',difference)
###The difference matrix successfully stores negative values. We rescale it to [0,255] to show the uint8 type image.
min_pixel = np.min(difference)
max_pixel = np.max(difference)
for r in range(difference.shape[0]):
    for c in range(difference.shape[1]):
            difference[r][c] = np.round(255*(difference[r][c]-min_pixel)/(max_pixel-min_pixel))
difference = np.uint8(difference)#imwrite always expects [0,255] range
cv2.imwrite('output\ps1-4-d-1.png',difference)
###Another way to scale
###difference = exposure.rescale_intensity(difference, out_range=(0, 255))


#Noise
##Take the original img1 and add Gaussian noise to the pixels in the green channel.
##Increase sigma until the noise is somewhat visible.
img1_with_green_noise = np.int16(image1.copy())
mu, sigma = 0, 10 # mean and standard deviation of Gaussian distribution
noise = np.round(np.random.normal(mu, sigma, img1_with_green_noise[:,:,1].shape))


###After adding (+/-) noise to green channel, the pixel values clips at 0 and 255
img1_with_green_noise[:,:,1] = np.int16(img1_with_green_noise[:,:,1]) + noise
###Truncate at 0 and 255
ret,img1_with_green_noise[:,:,1] = cv2.threshold(img1_with_green_noise[:,:,1],255,255,cv2.THRESH_TRUNC)
ret,img1_with_green_noise[:,:,1] = cv2.threshold(img1_with_green_noise[:,:,1],0,0,cv2.THRESH_TOZERO)
img1_with_green_noise = np.uint8(img1_with_green_noise)
cv2.imwrite('output\ps1-5-a-1.png',img1_with_green_noise)


##Instead add that amount of noise to the blue channel. The pixel values clips at 0 and 255
img1_with_blue_noise = np.int16(image1.copy())
img1_with_blue_noise[:,:,0] = np.int16(img1_with_blue_noise[:,:,0]) + noise
###Truncate at 0 and 255
ret,img1_with_blue_noise[:,:,0] = cv2.threshold(img1_with_blue_noise[:,:,0],255,255,cv2.THRESH_TRUNC)
ret,img1_with_blue_noise[:,:,0] = cv2.threshold(img1_with_blue_noise[:,:,0],0,0,cv2.THRESH_TOZERO)
img1_with_blue_noise = np.uint8(img1_with_blue_noise)
cv2.imwrite('output\ps1-5-b-1.png',img1_with_blue_noise)

