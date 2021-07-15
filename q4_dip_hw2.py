
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

originalImage = cv2.imread('/content/HeadCT.tif',0)

def transform2(img, A , B):
  img_shape = img.shape
  result = np.zeros(img_shape)
  height = img_shape[0]
  width = img_shape[1]
  for row in range(width):
      for column in range(height):
          if img[column, row] > B:
            result[column, row] = 0
          elif img[column, row] < A:
            result[column, row] = 0
          else :
            result[column, row] = img[column, row] 
  return result

newimg = transform2(originalImage ,51,200) # 51 and 200 are the reason of trial and error

fig, axs = plt.subplots(1, 2)

axs[0].imshow(originalImage, cmap="gray" )
axs[0].set_title('original Image');axs[0].axis('off')
axs[1].imshow(newimg, cmap="gray")
axs[1].set_title('new image');axs[1].axis('off')

plt.subplots_adjust(bottom = 0.025 , hspace =0.5)

def plotting(A,B):
  f = []
  for i in range(255):
    if( i < A):
      f.append(0)
    elif( i > B):
      f.append(0)
    else:
      f.append(i)
  return f

plt.figure()
font = {'family': 'serif',
        'weight': 'normal',
        'size': 16,
        }
f = plotting(51,200)
plt.plot(f)
plt.title(' Plot of Function ')
plt.xlabel('r', fontdict=font)
plt.ylabel('f(r)' , fontdict=font)

