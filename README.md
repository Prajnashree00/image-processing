# image-processing<br>
import cv2<br>
img=cv2.imread('leaf1.jpg',0)<br>
cv2.imshow('leaf1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/97970956/174995803-db53dd6a-14fe-426c-9dcd-6cf0466c102e.png)<br>



1.cover the image to URL code<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://i.natgeofe.com/n/3861de2a-04e6-45fd-aec8-02e7809f9d4e/02-cat-training-NationalGeographic_1484324_3x2.jpg'<br>
image=io.imread(url)
plt.imshow(image)<br>
plt.show()<br>
<br>
output:<br>
![image](https://user-images.githubusercontent.com/97970956/175013222-2b97c85a-cc09-4285-9763-08249fb4c856.png)<br>



import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('ow.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/175023065-853033e8-ec58-46ad-b5c6-2ba27e546234.png)<br>


hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
ithubusercontent.com/97970956/175023157-0320f0f4-eee6-4aa1-bd66-4c1aba56febb.png)<br>
<br>


light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97970956/175023247-8a5cb492-e995-4afa-9c40-2d757a151820.png)<br>


final_mask=mask+mask_white<br><br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()
![image](https://user-images.githubusercontent.com/97970956/175023337-725d8500-558c-4dee-b78a-28862519f347.png)
<br>
blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/175023457-d139c3cf-e08e-428f-9951-74a154921e79.png)<br>

3.write a program to perform arithmatic operation on images?<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

#reading image files<br>
img1=cv2.imread('doll1.jpg')<br>
img2=cv2.imread('doll2.jpg')<br>

#applying numpy additional on  images<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

#Saving the output image<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg4)<br>

![image](https://user-images.githubusercontent.com/97970956/175269788-5c557bb5-7eba-4cec-bc0d-5d69bd2fcedf.png)<br>
![image](https://user-images.githubusercontent.com/97970956/175269863-c4a4d5ed-f2c5-4b3a-be7c-f4a7b746a45e.png)<br>
![image](https://user-images.githubusercontent.com/97970956/175269989-6c895a23-ab66-4106-990d-680225907fc5.png)<br>
![image](https://user-images.githubusercontent.com/97970956/175270039-07787229-6968-4350-836d-bd8a47a07761.png)<br>

4.write a program to image to different color space<br>
import cv2<br>
img=cv2.imread("D:\\red.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
 output:<br>
 ![image](https://user-images.githubusercontent.com/97970956/175275071-22559b5a-5e21-458b-9c40-02a28233858e.png)<br>
![image](https://user-images.githubusercontent.com/97970956/175275144-56465b75-4b49-4bd7-8d68-1f703c9d071e.png)<br>
![image](https://user-images.githubusercontent.com/97970956/175275210-2ff7887e-4c65-4def-8918-4e3f8aabccc9.png)<br>
![image](https://user-images.githubusercontent.com/97970956/175275307-444c32ed-c450-4562-ab85-17dbcd7dfc07.png)<br>
![image](https://user-images.githubusercontent.com/97970956/175275393-1d1bc373-722a-40e3-a4de-ee59a077519c.png)<br>

5.program to create an image using 2Darray<br>

import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([255,130,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>

![image](https://user-images.githubusercontent.com/97970956/175284080-290e0547-715f-4db4-a9a1-62865938ed35.png)<br>

6.Image processing using bitwise operator?<br><br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('nature.jpg')<br>
image2=cv2.imread('nature.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and (image1,image2)<br>
bitwiseOr=cv2.bitwise_or (image1,image2)<br>
bitwiseXor=cv2.bitwise_xor (image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not (image1,image2)<br>
bitwiseNot_img2=cv2.bitwise_not (image1,image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

Outpu:<br>

![image](https://user-images.githubusercontent.com/97970956/176402963-f4adb2c4-7d5c-435b-9aac-1543138370d5.png)


7.blur image<br>
import cv2<br>
import numpy as np<br>

image=cv2.imread('puppy2.jpg')<br>

cv2.imshow('original Image',image)<br>
cv2.waitKey(0)<br>

#gaussian blur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>

#median Blur<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>

#Bilateral Blur<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
