   
import cv2
import numpy as np 
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
import time
import model.alpr as alpr


from ISR.models import RRDN


# from isr2 import sr



lpr = alpr.AutoLPR(decoder='beamSearch', normalise=True)
lpr.load(crnn_path='model/weights/best-fyp-improved.pth')






def resizetoh(img,height):
    r = height / img.shape[1]
    dim = (height, int(img.shape[0] * r))
    img = cv2.resize(img,dim,interpolation=cv2.INTER_CUBIC)
    return img



def zoom(img,r):

    dim = (int(img.shape[1]/(2*r)), int(img.shape[0]/(2*r)))
    center = (int(img.shape[1]/2), int(img.shape[0]/2))


    top= center[0]-dim[0]
    left = center[1]-dim[1]
    bottom= center[0]+dim[0]
    right = center[1]+dim[1]

    img = img[left:right,top:bottom]
    return img




def rotateimg(img,angle=None):

    if angle is None:
        angle = determine_skew(img)

    if abs(angle)>45:
        img = rotate(img, abs(angle)/angle*(90-abs(angle)), resize=False,cval=1) * 255
    else:
        img = rotate(img,angle, resize=False,cval=1) * 255

    return img.astype(np.uint8),angle




def preprocessing(image):

    # grayscale = rgb2gray(image)
    
    # grayscale = cv2.GaussianBlur(grayscale,(7,7),0)


    # rotated = rotate(image,0, resize=False) * 255
    # rotated = cv2.bitwise_not(rotated.astype(np.uint8))
    
    # height = 1000
    # r = height / rotated.shape[1]
    # dim = (height, int(rotated.shape[0] * r))

    # resized = cv2.resize(rotated,dim,interpolation=cv2.INTER_CUBIC)
    # kernel = np.ones((1, 1), np.uint8)
    # dialated = cv2.dilate(resized,kernel)
    # # cv2.imshow("DIA1",dialated)

    # img = cv2.cvtColor(dialated,cv2.COLOR_RGB2GRAY)

    # # _,img = cv2.threshold(diagrey,240,255,cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 75, 8)

    # img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

    # img = zoom(img,1.1)
    # img = np.where(img<200,0,img)
    # x = lpr.predict(img)
    # cv2.putText(img, x, (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
    # cv2.imshow("th",img)

    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.dilate(img,kernel)

    image = cv2.bitwise_not(image)
    img = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)




    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)#increases the white region in the image 
    img = cv2.erode(img, kernel, iterations=1) #erodes away the boundaries of foreground object

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



    # img,_ = rotateimg(img)

    img = np.stack((img,)*3,axis=-1)
    # cv2.imshow('pp',img)
    return img






def non_max_suppression_fast(boxes, overlapThresh):
    
	if len(boxes) == 0:
		return []
        
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
        
	pick = []
    
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
    
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
    
	while len(idxs) > 0:
        
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
        
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
        
		overlap = (w * h) / area[idxs[:last]]
        
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
            
            
	return boxes[pick].astype("int")


def fineturn(bboxes):
    
    widvec = bboxes[:,2]-bboxes[:,0]
    median = np.median(widvec)

    idx = np.where(widvec>1.5*median)[0]


    for i in idx:

        left = bboxes[i,0]
        top = bboxes[i,1]
        right = bboxes[i,2]
        bottom = bboxes[i,3]

        b1 = np.array([left,top,(right+left)//2,bottom]).reshape(-1,4)
        b2 = np.array([(right+left)//2,top,right,bottom]).reshape(-1,4)

        bboxes[i] = b1
        bboxes = np.vstack([bboxes,b2])

    return bboxes


counter = 0
def borderimg(img):
    img = cv2.rectangle(img,(1,1),(img.shape[1]-1,img.shape[0]-1),(255,255,255),1)

    return img


def test(im):
    global counter
    kernel = np.ones((4, 4), np.uint8)





    def refineimg(img,fbb):
        backgrnd = np.ones(img.shape)
        
        for x in fbb:
            cropped = img[x[1]:x[3],x[0]:x[2]]
            cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            backgrnd[x[1]:x[3],x[0]:x[2]] = cv2.addWeighted(backgrnd[x[1]:x[3],x[0]:x[2]],0, cropped,1, 0,dtype = cv2.CV_32F)

        return backgrnd



    def borderimg(img):
        img = cv2.rectangle(img,(1,1),(img.shape[1]-1,img.shape[0]-1),(255,255,255),1)

        return img


    def inlimits(w,h):

        if h/w>=1 and h/w<10:
            return 1
        return 0















    height = 1000
    width = 400
    r = height / im.shape[1]
    # dim = (height, int(im.shape[0] * r))
    dim = (height, width)
    im = cv2.resize(im,dim,interpolation=cv2.INTER_CUBIC)




    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('b',im)
    # im = cv2.bitwise_not(im)
    # cv2.imshow('a',im)

    blur = cv2.GaussianBlur(im,(3,3),0)
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 75, 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



    kernel = np.ones((4, 4), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=4)



    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)





    # cv2.imshow('a',thresh)

    
    # thresh,angle= rotateimg(thresh)
    thresh = borderimg(thresh)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]

    finalbb = []
    for cnt in contours:
        if cv2.contourArea(cnt)>2000:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  inlimits(w,h):
                finalbb.append([x,y,x+w,y+h])

    


    fbb = non_max_suppression_fast(np.array(finalbb),0.99)
    fbb = fineturn(np.array(fbb))


    # im,_ = rotateimg(im,angle=angle)
    for i in fbb:
        imx = np.stack((im,)*3,axis=-1)
        try:
            cv2.imwrite('chars/{}.jpg'.format(counter),thresh[i[1]-7:i[3]+7,i[0]-7:i[2]+7])
        except:
            cv2.imwrite('chars/{}.jpg'.format(counter),thresh[i[1]:i[3],i[0]:i[2]])
        counter+=1
    

        


    thresh = refineimg(im,fbb)

    # thresh,_ = rotateimg(thresh)

    im = np.stack((thresh,)*3,axis=-1)

    # for i in fbb:
    #     cv2.rectangle(im,(i[0],i[1]),(i[2],i[3]),(0,0,0),2)
    

    return im



def gettextarea(img):

    def getorigdim(var,refrom,reto):
        return(var*refrom//reto)



    orig = img.copy()


    oshape = orig.shape
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resizetoh(img,1000)
    ishape = img.shape


    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    
    kernel = np.ones((1, 50), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)

    img = borderimg(img)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]
    
    finalbb = []
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt)>1000 and cv2.contourArea(cnt)<400000 and w>h:            
            finalbb.append([x,y,x+w,y+h])

    fbb = non_max_suppression_fast(np.array(finalbb),0.99)

    for i in fbb:
        cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(0,0,0),2)

    


    fbbn = np.array(fbb)
    

    hmax = getorigdim(max(fbbn[:,3]-fbb[:,1]),oshape[1],ishape[1])

    fimg = np.ones((hmax,1,3),dtype=np.uint8)
    
    fbbn = np.array(sorted(fbbn,key=lambda x:x[0]))

    for box in fbbn:
        left = getorigdim(box[0],oshape[0],ishape[0])
        top = getorigdim(box[1],oshape[1],ishape[1])
        right = getorigdim(box[2],oshape[1],ishape[1])
        bottom = getorigdim(box[3],oshape[0],ishape[0])

        i1 = orig[top:bottom,left:right,:]
        
        
        i1 = cv2.copyMakeBorder(i1,0,hmax-(bottom-top),0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
        
        fimg = cv2.hconcat([fimg,i1])

    return fimg



def pad(f):
    x = 0.02
    f[0] = int(f[0]*(1-x))
    f[1] = int(f[1]*(1))
    f[2] = int(f[2]*(1+x))
    f[3] = int(f[3]*(1+x))

    return f

def show(img,a='a'):
    cv2.imshow(a,img)
    cv2.waitKey(0)

i = 0
rdn = RRDN(weights='gans')
def extex(img,bboxes):
    global i,rdn
    texts = []
    for box in bboxes:
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        

        # img3 = preprocessing(np.array(img[top:bottom,left:right,:]))
        


        if abs((left-right)/(bottom-top))<=2:
            box = pad(box)
            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]          
            # p = splitjoin(img[top:bottom,left:right,:],abs(top-bottom))
            p = gettextarea(img[top:bottom,left:right,:])

        else:
            p = img[top:bottom,left:right,:]
            
        img3 = np.array(p)
        
        # show(img3)

        
        


        # img3 = resizetoh(img3,1000)


        # img3 = cv2.GaussianBlur(img3,(3,3),0)


        
        # img3 = sr(img3)

        


        # kernel = np.ones((2, 2), np.uint8)
        # img3 = cv2.dilate(img3, kernel, iterations=3)





        img3 = rdn.predict(img3)


        cv2.imwrite('./olp/{}.jpg'.format(i),img3)
        i+=1
        # cv2.waitKey(0)

        x = lpr.predict(img3)[0]
        print(x)
        texts.append(x)

        # cv2.waitKey(0)
    return texts


if __name__=="__main__":
    gettextarea(cv2.imread('3.png'))
