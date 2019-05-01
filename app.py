from flask import Flask, render_template,Response
import cv2
from keras import models
import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

model = models.load_model("agemodel.h5")
model2 = models.load_model("gendermodel.h5")
model3 = models.load_model("racemodel.h5")
label1=['male','female']
label2=['White', 'Black', 'Asian', 'Indian','Hispanic']

fd = cv2.CascadeClassifier(r"haarcascadefiles/haarcascade_frontalface_alt.xml")

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
def get_face2(img):
    corners = fd.detectMultiScale(img,1.3,4)
    if len(corners)==0:
        return None,None 
    else:
        (x,y,w,h)= corners[0]
        img = img[x:x+h,y:y+w]   # cropping the image and resize them,
        img = cv2.resize(img,(100,100))
        return (x,y,w,h),img
    
    
def get_cosmetics(gender,race):
    if gender=="male" and race=="white":
        return "product1"
    elif gender=="female" and race=="white":
        return "product2"
    if gender=="male" and race=="Black":
         return "product3"
    elif gender=="female" and race=="Black":
        return "product4"
    else:
        return "product5"
def gen():
    vid = cv2.VideoCapture(0)
    while True:
        ret,img = vid.read()
        #img2=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corner,img2=get_face2(img)
        if corner !=None:
            with graph.as_default():
                age_out= model.predict(img2.reshape(1,100,100,3))
                gender_out= model2.predict_classes(img2.reshape(1,100,100,3))[0][0]
                race_out= model3.predict_classes(img2.reshape(1,100,100,3))[0]
            #emotion=labels[output[0]]
            text = str(age_out[0][0])+' '+label1[gender_out]+' '+label2[race_out]
            text2 =  " product " + get_cosmetics(gender_out,race_out)
            (x,y,w,h) = corner
            cv2.putText(img,text,(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
            cv2.putText(img,text2,(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        ret,jpeg = cv2.imencode(".jpg",img)
        yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
    