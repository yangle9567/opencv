# 导入工具包
from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 设置参数 --image credit_card_03.png --template ocr_a_reference.png
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
                help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}
# 绘图展示
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 读取一个模板图像
img = cv2.imread(args["template"])
# cv_show('img',img)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('ref',ref)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
# cv_show('ref',ref)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓

refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show('img-contours',img)
print (np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts,method="left-to-right")[0]
digits = {}

for (i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y+h,x:x+w]
    roi = cv2.resize(roi,(57,88))
    digits[i] = roi

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

image = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image,width=300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show('tophat',tophat)

gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradX = np.absolute(gradX)

(minVal,maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gradX',gradX)

gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
cv_show('gradX',gradX)

thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
cv_show('thresh',thresh)

threshCnts, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)
locs = []

#遍历轮廓
for (i,c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w / float(h)

    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x,y,w,h))

locs = sorted(locs,key=lambda x:x[0])
output = []

#遍历每一个轮廓中的数字
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []

    group = gray[gY - 5:gY + gH + 5,gX - 5:gX + gW + 5]
    cv_show('group',group)
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group',group)
    digitCnts,hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
    for c in digitCnts:
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y + h,x:x + w]
        roi = cv2.resize(roi,(57,88))
        cv_show('roi',roi)
        scores = []

        for (digit,digitRIO) in digits.items():
            result = cv2.matchTemplate(roi,digitRIO,cv2.TM_CCOEFF)
            (_,score,_,_) = cv2.minMaxLoc(result)
            scores.append(score)

        groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(image,(gX - 5,gY - 5),(gX + gW + 5,gY + gH + 5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY - 15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    output.extend(groupOutput)

print("Credit Card Type :{}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
























