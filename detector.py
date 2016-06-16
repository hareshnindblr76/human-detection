import cv2
import os,sys
import numpy as np
#import dlib
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = {'winStride': (8, 8), 'padding': (16, 16), 'scale': 1.03}
#f_detector = cv2.CascadeClassifier('C:/opencv-3.1/opencv/data/haarcascades/haarcascade_fullbody.xml')
structured_edges_file='./models/model.yml.gz'
#fzb = cv2.ximgproc.segmentation.createGraphSegmentation() #Tried segmengtation...it was slow
def normalize_image(img, window_size):
	image = img.astype(np.float32)
	if window_size>0:
		for i in range(0, img.shape[0],window_size):
			for j in range(0, img.shape[1],window_size):
				for ch in range(3):
					y = min(img.shape[0],i+window_size)
					x = min(img.shape[1], j+window_size)
					maxim = img[i:y,j:x,ch].max()
					image[i:y, j:x, ch] /=maxim
	return image
	
def detect_edges(image, structured_edges=True):
	edges2 = np.zeros(image.shape, dtype=np.float32)
	if image.max()!=0:
		try:
			edge_detector = cv2.ximgproc.createStructuredEdgeDetection(structured_edges_file) 

			edges2 = edge_detector.detectEdges( image.astype(np.float32) );
			edges2*=255.0
			edges2=edges2.astype(np.uint8)
		except:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			#gray = cv2.GaussianBlur(gray, (3,3),1.0)
			
			edgesX = cv2.Sobel(gray, cv2.CV_32F, 1,0,5, cv2.BORDER_REPLICATE)
			edgesY = cv2.Sobel(gray, cv2.CV_32F,0,1,5,cv2.BORDER_REPLICATE)
			
			edges = np.sqrt(edgesX**2 + edgesY**2)
			edges /= edges.max()
			edges *= 255.0
			edges2 = edges.astype(np.uint8)
	return edges2
def apply_morphology(edges2, threshVal=40):

	if threshVal>0:
		_,edges2 = cv2.threshold(edges2, threshVal,255,0)
		edges2 = cv2.erode(edges2, (3,3))
	morph = cv2.morphologyEx(edges2, cv2.MORPH_OPEN, (3,11),iterations=2)
	morph = cv2.morphologyEx(morph,cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,35)), iterations=1) 
	return morph	
#Assuming single human in images.
def detect_humans(img_files):
	for img_file in img_files:

		img = cv2.imread(img_dir+img_file)

		resiz = True
		if resiz:

			img = cv2.resize(img, (400,400),0,0,cv2.INTER_AREA) #To handle scale issues as  HOG didn't detect on large images.Seems to work well on the given dataset for the HOG params and morpho operations.
		img = cv2.GaussianBlur(img, (5,5),1.5,1.0)
		image  = img.astype(np.float32)/255.0
		'''
		image = normalize(img,40)
		'''
					
		#print "normed out",image.max()

		'''

		'''
		edges2 = detect_edges(image)
		thresh = apply_morphology(edges2)
		thresh_e = cv2.erode(thresh,(3,3))
		
		#ret, thresh = cv2.threshold(edges2, 30, 255,0)
		_,contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


		max_area = 0
		max_index = -1
		i=0
		detections = []
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			asp = float(w)/h
			# The model in the photo should occupy atleast 5% of the image
			if w*h>(img.shape[0]*img.shape[1])/20 and  asp > 0.2 and asp < 1.2:
				detections.append([x,y,w,h])
				if w*h>max_area:
					max_area=w*h
					max_index=i
				i+=1
				
				#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		

		rects,_ = hog.detectMultiScale(edges2, **hogParams)
		#rects = pedestrian_detector.detectMultiScale(gray, 1.2,3)
		
		
		for x,y,w,h in rects:
			if w*h>(img.shape[0]*img.shape[1])/20 and  asp > 0.2 and asp < 1.2:
				detections.append([x,y,w,h])
				if w*h>max_area:
					max_area=w*h
					max_index=i
				i+=1
			
			#cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,0),2)
		sec_max_area = 0
		x,y,w,h = detections[max_index]
		cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
			
		'''
		Tried out Maximally stable extremal regions....didn't work good :\
		mser = cv2.MSER_create(_delta=20, _min_area=1000, _max_area=100000)
		_,rects = mser.detectRegions(img)
		for x,y,w,h in rects:
			cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)
		'''
		#cv2.imshow("edges",thresh_e)
		#cv2.imshow("sobel",edges)
		#cv2.imshow("edgesY", np.absolute(edgesY).astype(np.uint8))
		#cv2.imshow("boxes",img)
		if not os.path.isdir(img_dir+'/detections/'):
			os.mkdir(img_dir+'/detections/')
		cv2.imwrite(img_dir+'/detections/'+img_file[:-4]+"_box_detections.png",img)
		#cv2.imshow("fgmsk",ycrcb)
		
		#cv2.waitKey(0)
		#break
		
	#cv2.destroyAllWindows()
def get_img_files(im_dir, extensions=['.jpg','.jpeg']):

	img_files = [f for f in os.listdir(im_dir) if any( [f.endswith(e) for e in extensions])]
	return img_files

def write_pos_img_files(img_files):
	outfile = open("img_files.txt","w")
	for img_file in img_files:
		outfile.write(img_file+'\n')
	outfile.close()
if __name__ == '__main__':
	img_dir = sys.argv[1]
	
	extensions = ['.jpg','.jpeg']
	if len(sys.argv )> 2:
		extensions = sys.argv[2:]
	#img_dir = "./" #"../PennFudanPed/PNGImages/"
	
	#structured_edges_model = 'model.yml.gz'

	img_files = get_img_files(img_dir, extensions)
	detect_humans(img_files)

