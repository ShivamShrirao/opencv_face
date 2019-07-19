import cv2

face_cascade = cv2.CascadeClassifier('/usr/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')

cam = cv2.VideoCapture(0)

while True:
	ret, img = cam.read()
	img = cv2.flip(img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		eye_gray = gray[y:y+h,x:x+w]
		eye_col = img[y:y+h,x:x+w]
		eyes = eye_cascade.detectMultiScale(eye_gray)
		for (x,y,w,h) in eyes:
			cv2.rectangle(eye_col, (x,y), (x+w,y+h),(0,255,0),2)

	cv2.imshow('webcam', img)
	if cv2.waitKey(1) & 0xff == 27:
		break

cam.release()
cv2.destroyAllWindows()