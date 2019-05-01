import numpy
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im

import os
folder = r"part3"
files = os.listdir(folder)
files

fd = cv2.CascadeClassifier(  r"haarcascadefiles\haarcascade_frontalface_alt.xml"  )
def get_face(path):
    img=cv2.imread(path)
    corners = fd.detectMultiScale(img,1.3,4)
    if len(corners)==0:
        return None,None 
    else:
        (x,y,w,h)= corners[0]
        img = img[x:x+h,y:y+w]   # cropping the image and resize them,
        img = cv2.resize(img,(100,100))
        return (x,y,w,h),img
    
trainimg = []
trainlb_age = []
trainlb_gender = []
trainlb_race = []
for file in files:
    if ".jpg" in file:
        path = folder+'\\'+file 
        details = file.split("_")
        age = details[0]
        gender = details[1]
        race = details[2]
        if int(age)>18 and int(age)<50:
            try:
                corner,img = get_face(path)
                if corner !=None:
                    trainimg.append(img)
                    trainlb_age.append(int(age))
                    trainlb_gender.append(int(gender))
                    trainlb_race.append(race)
            except :
                print("failed")

trainimg = numpy.array(trainimg)
trainlb_age = numpy.array(trainlb_age).reshape(1893,1) 
trainlb_gender = numpy.array(trainlb_gender).reshape(1893,1) 
trainlb_race = numpy.array(trainlb_race)

trianimg = trainimg/255

#reshape the image data 
trainimg = trainimg.reshape(1893,100,100,3)  

from sklearn.preprocessing import OneHotEncoder
trainlb_race = OneHotEncoder().fit_transform(trainlb_race.reshape(1893,1)).toarray() # should excute once والا بيطلع ايرور

print(trainimg.shape)
print(trainlb_age.shape)
print(trainlb_gender.shape)
print(trainlb_race.shape)

from sklearn.metrics import mean_squared_error

from keras import models, layers 
model = models.Sequential()

# add first convolutional and maxpoling layer
model.add(layers.Conv2D(filters=20,kernel_size=(3,3),activation='relu',input_shape=(100,100,3))) # بدهاش تزبط يا زلمه
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# add the second convolutional layer
model.add(layers.Conv2D(filters=40,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# add the flatten layer
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(trianimg,trainlb_age,epochs=20,batch_size=50,shuffle =True, verbose = True)

model.save("agemodel.h5")

from keras import models, layers 
model2 = models.Sequential()

# add first convolutional and maxpoling layer
model2.add(layers.Conv2D(filters=20,kernel_size=(3,3),activation='relu',input_shape=(100,100,3)))
model2.add(layers.MaxPooling2D(pool_size=(2,2)))

# add the second convolutional layer
model2.add(layers.Conv2D(filters=40,kernel_size=(3,3),activation='relu'))
model2.add(layers.MaxPooling2D(pool_size=(2,2)))

# add the flatten layer
model2.add(layers.Flatten())
model2.add(layers.Dense(20,activation='relu'))
model2.add(layers.Dense(1,activation= 'sigmoid'))

model2.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

model2.fit(trianimg,trainlb_gender,epochs=10,batch_size=50,shuffle =True, verbose = True)

model2.save("gendermodel.h5")

print(trainimg.shape)
print(trainlb_race.shape)

from keras import models, layers 
model3 = models.Sequential()

# add first convolutional and maxpoling layer
model3.add(layers.Conv2D(filters=20,kernel_size=(3,3),activation='relu',input_shape=(100,100,3)))
model3.add(layers.MaxPooling2D(pool_size=(2,2)))

# add the second convolutional layer
model3.add(layers.Conv2D(filters=40,kernel_size=(3,3),activation='relu'))
model3.add(layers.MaxPooling2D(pool_size=(2,2)))

# add the flatten layer
model3.add(layers.Flatten())
model3.add(layers.Dense(20,activation='relu'))
model3.add(layers.Dense(5,activation= 'softmax'))

model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy'])
model3.fit(trianimg,trainlb_race,epochs=10,batch_size=50,shuffle =True, verbose = True)

model3.save("racemodel.h5")
