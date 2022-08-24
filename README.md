# image-processing<br>
1.develop a program to display grayscal;e image in using read and write operation<br>
import cv2<br>
img=cv2.imread('leaf1.jpg',0)<br>
cv2.imshow('leaf1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/97970956/174995803-db53dd6a-14fe-426c-9dcd-6cf0466c102e.png)<br>

2. Develop a program to display the image using matplotlib<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread("d1.jpg")<br>
plt.imshow(img) <br>
![image](https://user-images.githubusercontent.com/97970956/178448790-f2df7bb2-1c5e-43b7-afd0-e68889d85afe.png)<br>

3.develop a program to perform linear transformation rotation<br>
from PIL import Image<br>
img=Image.open("plant1.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/97970956/178449546-bdab8987-7134-4a12-85a6-32ca286d082c.png)<br>

4.Develop a program to convert color string RGB color values<br>
import cv2<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img1=ImageColor.getrgb("red")<br>
print(img1)<br>
img1=ImageColor.getrgb("pink")<br>
print(img1)<br>
OUTPUT:<br>
(255, 255, 0)<br>
(255, 0, 0)<br>
(255, 192, 203)<br>

5.Write a program to create image using color<br>
from PIL import Image)<br>
img=Image.new("RGB",(200,400),(255,255,0)))<br>
img.show())<br>
![image](https://user-images.githubusercontent.com/97970956/178453176-86d8a116-f0fd-44c1-b4ff-ebfe1bc9335a.png)<br>

6.develop a program to utilize the image using various color spaces<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('d1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/178453729-c1411bb9-88b7-458c-8836-b3c6b1679e1e.png)<br>

7.write a program to display image attributes<br>
from PIL  import Image<br>
image=Image.open('d1.jpg')<br>
print("filename:",image.filename)<br>
print("mode:",image.mode)<br>
print("size:",image.size)<br>
print("width:",image.width)<br>
print("height:",image.height)<br>
image.close()<br>
output:<br>
filename: d1.jpg<br>
mode: RGB<br>
size: (259, 194)<br>
width: 259<br>
height: 194<br>

8.Resize the original image<br>
import cv2<br>
img=cv2.imread('flower1.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('resized image lenght width',imgresize.shape)<br>
cv2.waitKey(0)<br>
![image](https://user-images.githubusercontent.com/97970956/178961065-cd576dac-e89a-43b3-aaf6-2218033eac35.png)<br>
![image](https://user-images.githubusercontent.com/97970956/178961238-f94602f1-5b69-497b-80b7-05d2e8357d97.png)<br>
![image](https://user-images.githubusercontent.com/97970956/178962689-f36d053a-dabb-4cca-877c-93967f5fa5ba.png)<br>


9.convert the original to grey scale and then to binary ?<br>
import cv2<br>

img=cv2.imread('flower3.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
#gray scale<br>
img=cv2.imread ('flower3.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
#binary image<br>
ret,bw_img=cv2.threshold (img,127,255,cv2.THRESH_BINAR<br>
Y)
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/97970956/176482938-0a07162e-6211-4901-843e-cee5665dac3a.png)
![image](https://user-images.githubusercontent.com/97970956/176483282-ae408cc6-4aba-4071-ab97-3ef3d37b15d3.png)
![image](https://user-images.githubusercontent.com/97970956/176485862-99fc51b5-5b4d-4b82-828b-aa9cf4a43462.png)


**Lab excercise**
**1.cover the image to URL code**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://i.natgeofe.com/n/3861de2a-04e6-45fd-aec8-02e7809f9d4e/02-cat-training-NationalGeographic_1484324_3x2.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
<br>
output:<br>
![image](https://user-images.githubusercontent.com/97970956/175013222-2b97c85a-cc09-4285-9763-08249fb4c856.png)<br>


**2.Write a program to mask and blur the image **<br>
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
![image](https://user-images.githubusercontent.com/97970956/176486243-2c7c3eca-2792-4257-aa19-e20b7084214d.png)



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

**3.write a program to perform arithmatic operation on images?**<br>
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

**4.write a program to image to different color space**<br>
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

**5.program to create an image using 2Darray**<br>

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

**6.Image processing using bitwise operator?**<br><br>
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


**7.blur image**<br>
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

ouput:<br>
![image](https://user-images.githubusercontent.com/97970956/176412977-4a5124d5-7417-40ec-b3da-e34a390204ae.png)
![image](https://user-images.githubusercontent.com/97970956/176413071-92949d59-4da3-4ec8-b2ab-3ea08b34761b.png)
![image](https://user-images.githubusercontent.com/97970956/176413147-57ad7de0-8034-4950-944f-bc2a7d3276db.png)
![image](https://user-images.githubusercontent.com/97970956/176413236-bcfff73d-abb0-4c2a-af1f-458ef07d302f.png)

**8.Image enhancement**<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('nature.jpg')<br>
image.show()<br>

enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightned=enh_bri.enhance(brightness)<br>
image_brightned.show()<br>

enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>

enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_col.enhance(contrast)<br>
image_contrasted.show()<br>

enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>

Output:<br>
![image](https://user-images.githubusercontent.com/97970956/176422262-b2664c58-1c7b-4c4c-b59b-21be1d2ff4b7.png)<br>
![image](https://user-images.githubusercontent.com/97970956/176422298-ecc3bbe9-8ab8-4f54-90a7-5ebaa9e3c242.png)<br>
![image](https://user-images.githubusercontent.com/97970956/176422339-573b6029-7a41-42a1-8d16-fa62244c6d83.png)<br>
![image](https://user-images.githubusercontent.com/97970956/176422371-ff1b3598-634c-48c2-8317-8468571c21cb.png)<br>
![image](https://user-images.githubusercontent.com/97970956/176422432-f577810e-5225-4947-a8cf-1287deda10f3.png)<br>

**9.morphological**
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot  as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('nature.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

![image](https://user-images.githubusercontent.com/97970956/176427745-8c34e52f-0184-443e-935c-3705bd87f9e1.png)<br>

**10.Develop a program to<br><br>
i)Read the image convert it nto grayscale image <br>
ii)Write (save) the grayscale image and<br>
iii)display the original image and gray scale image<br>**
import cv2<br>
OriginalImg=cv2.imread('rabbit.jpg')<br>
GrayImg=cv2.imread('rabbit.jpg',0)<br>
isSaved=cv2.imwrite('D:/i.jpg',GrayImg)
cv2.imshow('Display original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('the image is sucessfully saved')<br>
 output:<br>
 ![image](https://user-images.githubusercontent.com/97970956/178699383-558a5127-1e18-4176-970d-4f6d84f71fd4.png)<br>
 ![image](https://user-images.githubusercontent.com/97970956/178699440-b9894839-8548-49b2-8434-23b6ab1440ea.png)<br>

**11.graylevel slicing with background**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/178705576-221a33b6-63f0-4125-893c-e1bb95494684.png)<br>

**12.graylevel slicing without background**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('graylevel slicing without background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/178706784-ed99b011-f1be-4039-b9f0-cdfdf261283a.png)<br>

13.skimage<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('rabbit.jpg')<br>

_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )<br>
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count')<br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/178966197-cd4627bc-8856-4199-b391-92fc2e40bb4c.png)<br>

or <br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = io.imread('rabbit.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
image = io.imread('rabbit.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/178967175-45a95ca9-db6e-4536-b156-c4c4b4561b30.png)

#numpy<br>
import cv2<br>
import numpy as np<br>
img  = cv2.imread('rabbit.jpg',0)<br>
hist = cv2.calcHist([img],[0],None,[256],[0,256])<br>
plt.hist(img.ravel(),256,[0,256])<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/178966309-98a1e3c1-13c3-467d-9adc-d1dc90168f5e.png)<br>

#opencv<br>
import cv2  <br>
from matplotlib import pyplot as plt  <br>
img = cv2.imread('rabbit.jpg',0) <br>
histr = cv2.calcHist([img],[0],None,[256],[0,256]) <br>
plt.plot(histr) <br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/178966396-859b2414-74a9-4139-a49c-f0afd4605926.png)<br>

**14.program to perform basic image data analysis using thransformation** <br>
a.Image negative  <br>
b.log transformation  <br>
c.gamma correction <br>

%matplotlib inline <br>
import imageio <br>
import matplotlib.pyplot as plt <br>
import warnings  <br>
import matplotlib.cbook <br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) <br>
pic=imageio.imread('tiger.jpg') <br>
plt.figure(figsize=(6,6)) <br>
plt.imshow(pic); <br>
plt.axis('off'); <br>
![image](https://user-images.githubusercontent.com/97970956/179962279-e4c5f2b4-3bec-4245-9b4c-fe581764bbc5.png) <br>

**a.Image negative** <br>
negative=255-pic    #neg=[l-1]-img <br>
plt.figure(figsize=(6,6)) <br>
plt.imshow(negative); <br>
plt.axis('off'); <br>
![image](https://user-images.githubusercontent.com/97970956/179962367-2eead373-a248-4928-867e-24f781e7cfa8.png) <br>

**b.log transformation**  <br>
%matplotlib inline <br>

import imageio <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>

pic=imageio.imread('tiger.jpg')
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114]) <br>
gray=gray(pic) <br>
max_=np.max(gray) <br>

def log_transform(): <br> <br>
    return(255/np.log(1+max_))*np.log(1+gray) <br>
plt.figure(figsize=(5,5)) <br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray')) <br>
plt.axis('off'); <br>
![image](https://user-images.githubusercontent.com/97970956/179962541-822c2c53-2777-4040-a362-af86baa426bf.png) <br>

**c.gamma correction** <br>
import imageio <br>
import matplotlib.pyplot as plt <br>

#gamma encoding <br>
pic=imageio.imread('tiger.jpg') <br>
gamma=2.2 #gamma<1~Drak; gamma>1~Bright <br>

gamma_correction=((pic/255)**(1/gamma)) <br>
plt.figure(figsize=(5,5)) <br>
plt.imshow(gamma_correction) <br>
plt.axis('off'); <br>
![image](https://user-images.githubusercontent.com/97970956/179962702-ad64abd2-997d-4036-871a-dafec6209f6b.png) <br>

**15. Program to perform basic image manipulation: <br>
a) Sharpness <br>
b) Flipping <br>
c) Cropping <br>
**
**a) Sharpness <br>**
#image sharpen<br>
from PIL import Image<br>
from PIL  import ImageFilter<br>
import matplotlib.pyplot as plt<br>

#Load the image <br>
my_image=Image.open('puppy2.jpg')<br>

#Use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>

#Save the image <br>
sharp.save('D:/prajna/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/180189629-403c642f-5162-47e5-bb0c-a923fc1b1cc1.png)<br>


**b) Flipping **<br><br>
#image flip<br>
import matplotlib.pyplot as plt <br>

#laod the image<br>
img=Image.open('puppy2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

#use the flip function<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>

#save the image<br>
flip.save('D:/prajna/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/180189800-d1bff529-713a-4bd0-bd5f-a46cd2eb57e3.png)<br>


**c) Cropping **<br><br>
#Importing Image class from PIL module <br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
#opens a image in RGB mode<br>
im=Image.open('puppy2.jpg')<br>

#size of the image in pixels(size of original image<br>
#(this is not mandatory)<br>
width,height=im.size<br>

#cropped image of above dimension<br>
#(it will not change original image)<br>
im1=im.crop((280,100,800,600))<br>
<br>
#shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97970956/180190015-b6086975-02b4-45d1-8a1c-e5f3d6502a87.png)<br>

**#gradient image**<br>
 import numpy as np<br>
import matplotlib.pyplot as plt<br>

arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0, 0, 0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        #Find the distance to the center<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>

        #Make it on a scale from 0 to 1innerColor<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>

        #Calculate r, g, and b values<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        # print r, g, b<br>
        arr[y, x] = (int(r), int(g), int(b))<br>

plt.imshow(arr, cmap='gray')<br>
plt.show()<br>
<br>
![image](https://user-images.githubusercontent.com/97970956/180202203-0eef0b10-e855-4535-9360-8c799cdb1f58.png)<br>

<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
  
# Create an image as input:<br>
input_image = Image.new(mode="RGB", size=(400, 400),<br>
                        color="blue")<br>
  
# save the image as "input.png"<br>
#(not mandatory)<br>
#input_image.save("input", format="png")<br>
  
# Extracting pixel map:<br>
pixel_map = input_image.load()<br>
  
# Extracting the width and height<br>
# of the image:<br>
width, height = input_image.size<br>
z = 100<br>
for i in range(width):<br>
    for j in range(height):<br>
        <br>
        # the following if part will create<br>
        # a square with color orange<br>
        if((i >= z and i <= width-z) and (j >= z and j <= height-z)):<br>
            
            # RGB value of orange.<br>
            pixel_map[i, j] = (255, 165, 255)<br>
  
        # the following else part will fill the<br>
        # rest part with color light salmon.<br>
        else:<br>
            
            # RGB value of light salmon.<br>
            pixel_map[i, j] = (255, 160, 0)<br>
  
# The following loop will create a cross<br>
# of color blue.<br>
for i in range(width):<br>
    
    # RGB value of Blue.<br>
    pixel_map[i, i] = (0, 0, 255)<br>
    pixel_map[i, width-i-1] = (0, 0, 255)<br>
  
# Saving the final output<br>
# as "output.png":<br>
#input_image.save("output", format="png")<br>
plt.imshow(input_image)<br>
plt.show()  <br>
# use input_image.show() to see the image on the<br>
# output screen.<br>
![image](https://user-images.githubusercontent.com/97970956/180202393-01ac441f-15d4-443c-a0b0-3e38b72dc97a.png)<br>


**rgb<br>**
from PIL import Image<br>
import numpy as np<br>
w, h = 600, 600<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
data[300:400, 300:400] = [130, 255, 0]<br>
data[400:500, 400:500] = [0, 255, 170]<br>
data[500:600, 500:600] = [180, 255, 0]<br>
# red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
![image](https://user-images.githubusercontent.com/97970956/180202481-a6e4be60-9f9e-4334-a8c3-647d3b1dd3f2.png)<br>

**#image to matrrix<br>**
import matplotlib.image as image<br>
img=image.imread('puppy2.jpg')<br>
print('The Shape of the image is:',img.shape)<br>
print('The image as array is:')<br>
print(img)<br>

![image](https://user-images.githubusercontent.com/97970956/180202646-2e5a517d-a319-41fb-8769-158724cc637d.png)<br>
![image](https://user-images.githubusercontent.com/97970956/180202680-1f605cf5-33cc-4374-8536-82ca1336a098.png)<br>
<br>

# example of pixel normalization<br>
from numpy import asarray<br>
from PIL import Image<br>
# load image<br>
image = Image.open('rabbit.jpg')<br>
pixels = asarray(image)<br>
# confirm pixel range is 0-255<br>
#print('Data Type: %s' % pixels.dtype)<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
# convert from integers to floats<br>
pixels = pixels.astype('float32')<br>
# normalize to the range 0-1<br>
pixels /= 255.0<br>
# confirm the normalization<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>

output:<br>
![image](https://user-images.githubusercontent.com/97970956/181229783-c0281e9a-87bb-44be-b9e8-97460e998790.png)<br>
#max<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('rabbit.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>

print(max_channels)<br>
![image](https://user-images.githubusercontent.com/97970956/181229918-34ff548e-d8d9-4558-8c2b-454aa1621d10.png)<br>

#min<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('rabbit.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>

print(min_channels)<br>
![image](https://user-images.githubusercontent.com/97970956/181230458-2413e504-3170-4668-9b2b-914fa11fe1db.png)<br>



#average<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread("rabbit.jpg")<br>
plt.imshow(img)<br>
np.average(img)<br>
![image](https://user-images.githubusercontent.com/97970956/181230295-2da1858d-6982-404e-9a32-dfcd776321cf.png)<br><br>

from PIL import Image,ImageStat<br>
import matplotlib.pyplot as plt<br>
im=Image.open('rabbit.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>
![image](https://user-images.githubusercontent.com/97970956/181230572-86b10a17-92a1-40d5-a9b4-5eb6ec43eee6.png)<br>

# Python3 program for printing<br>
# the rectangular pattern<br>
# Function to print the pattern<br>
def printPattern(n):<br>
 
    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
         
    # Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
             
    # Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 
# Driver Code<br>
n = 4;<br>
 
printPattern(n);<br>

![image](https://user-images.githubusercontent.com/97970956/181233899-5ebaf44f-8f12-448b-b499-652e1ff6cdf8.png)


#minimum<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('rabbit.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>

print(min_channels)<br>
![image](https://user-images.githubusercontent.com/97970956/186389190-4b6f3e56-4f3d-464b-83e3-133db34667b0.png)<br>
#maximum<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('rabbit.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>

print(max_channels)<br>
![image](https://user-images.githubusercontent.com/97970956/186389365-dff4bba9-5b5f-4cb5-8c00-75537c19ee3c.png)<br>



import imageio<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread("rabbit.jpg")<br>
plt.imshow(img)<br>
np.average(img)<br>

![image](https://user-images.githubusercontent.com/97970956/186389510-736f8eea-63d8-4637-8942-10fb1e6ea423.png)<br>

#standard deviation<br>
from PIL import Image,ImageStat<br>
import matplotlib.pyplot as plt<br>
im=Image.open('rabbit.jpg')<br>
plt.imshow(im)<br>
plt.show()
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>
![image](https://user-images.githubusercontent.com/97970956/186389669-1d87c6b8-5bb4-4a0e-adf7-89d327976174.png)<br>


Edge Detection Using OpenCV<br>
import cv2
2
 
3
# Read the original image
4
img = cv2.imread('rabbit.jpg')
5
# Display original image
6
cv2.imshow('Original', img)
7
cv2.waitKey(0)
8
 
9
# Convert to graycsale
10
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
11
# Blur the image for better edge detection
12
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
13
 
14
# Sobel Edge Detection
15
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
16
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
17
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
18
# Display Sobel Edge Detection Images
19
cv2.imshow('Sobel X', sobelx)
20
cv2.waitKey(0)
21
cv2.imshow('Sobel Y', sobely)
22
cv2.waitKey(0)
23
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
24
cv2.waitKey(0)
25
 
26
# Canny Edge Detection
27
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
28
# Display Canny Edge Detection Image
29
cv2.imshow('Canny Edge Detection', edges)
30
cv2.waitKey(0)
31
 
32
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97970956/186399087-6100775f-c900-40c5-a7e0-31e3b29e512a.png)
![image](https://user-images.githubusercontent.com/97970956/186399207-5bade395-8795-4b05-b27f-50c77f6a468c.png)
![image](https://user-images.githubusercontent.com/97970956/186399286-762c2f1e-18fd-46f0-bf12-2b59d62c935f.png)
![image](https://user-images.githubusercontent.com/97970956/186399388-c900a50f-af2f-4c73-8d30-b30b239a6a01.png)
![image](https://user-images.githubusercontent.com/97970956/186399454-b9a4dce8-5c5d-4b29-bd8b-614fb9cd0cd9.png)

# Sobel Edge Detection
2
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
3
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
4
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
5
 
6
# Display Sobel Edge Detection Images
7
cv2.imshow('Sobel X', sobelx)
8
cv2.waitKey(0)
9
 
10
cv2.imshow('Sobel Y', sobely)
11
cv2.waitKey(0)
12
 
13
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
14
cv2.waitKey(0)
