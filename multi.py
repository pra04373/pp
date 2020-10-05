import threading
import time
import cv2
import numpy as np
from keras.models import load_model
from keras_vggface import utils
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
image_size = 224


class myThread (threading.Thread):
    def __init__(self, src):
        print("thread -------------init-------------")
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(src)
        self.stop = False
    def run(self):
        while(self.stop == False):
            self.ret, self.frame = self.cap.read()

    def Stop(self):
        self.cap.release()
        self.stop = True

    def read(self):
        return self.ret, self.frame

model = load_model('/home/pkumar/Desktop/livevideo/vgg_2.h5')
HumanNames = ['nickole','praveen']
cascade_classifier = cv2.CascadeClassifier('/home/pkumar/Downloads/haarcascade_frontalface_default.xml')
# faceCascade = cv2.CascadeClassifier(cascadePath);

thread = myThread(0)
thread.start()
time.sleep(1)

start = time.time()
frames = 0
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(1)
while(True):
    ret, frame = thread.read()
    frame = cv2.resize(frame, (640, 480))
    frames += 1
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color = frame [y:y+h, x:x+w]
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        roi_color = cv2.resize(roi_color, (image_size, image_size))
        image = roi_color.astype(np.float32, copy=False)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image) # or version=2
        image /= 255
        preds = model.predict(image)
        print(preds)
        best_class_indices = np.argmax(preds, axis=1)
        best_class_probabilities = preds[np.arange(len(best_class_indices)), best_class_indices]
        print(best_class_indices,' with accuracy ',best_class_probabilities)
        if best_class_probabilities>0.75:
            for H_i in HumanNames:
                if HumanNames[best_class_indices[0]] == H_i:
                    result_names = HumanNames[best_class_indices[0]]
                    print(result_names)
                    cv2.putText(frame, result_names, (x, y ), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 255,0), thickness=1, lineType=2)



    # for(x,y,w,h) in faces:
    #     # Create rectangle around the face
    #     cv2.rectangle(frame, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
      
    #     # cv2.rectangle(frame, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
    #     # cv2.putText(frame, str(Id), (x,y-40), font, 2, (255,255,255), 3)  

    if cv2.waitKey(10) & 0xFF == ord('q'):
        thread.Stop()
        break
    cv2.imshow("frame", frame) 
end = time.time()
second = end - start
print("second:", + second)
print(frames/second)
cv2.destroyAllWindows()
