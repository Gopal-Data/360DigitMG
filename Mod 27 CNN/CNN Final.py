from matplotlib import pyplot
from keras.datasets import cifar10    
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
 
(trainX, trainY), (testX, testY) = cifar10.load_data() # loading the cifar data
trainY = to_categorical(trainY) #converting to binary
testY = to_categorical(testY) #converting to binary
	
trainX = trainX.astype('float32')
testX = testX.astype('float32')
 
# normalize to range 0-1
trainX = trainX / 255.0
testX = testX / 255.0

#Model with Extra Layers and Batch Normalizartion and dropout between Layer
model_norm = Sequential()
model_norm.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))#Filter layer
model_norm.add(BatchNormalization()) #Normalizing the layer
model_norm.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_norm.add(BatchNormalization()) #Normalizing the layer
model_norm.add(MaxPooling2D((2, 2))) #Maxpool on the layer
model_norm.add(Dropout(0.2))#20% Value Drop

model_norm.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_norm.add(BatchNormalization()) #Normalizing the layer
model_norm.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_norm.add(BatchNormalization()) #Normalizing the layer
model_norm.add(MaxPooling2D((2, 2)))#Maxpool on the layer
model_norm.add(Dropout(0.3))#30% Value Drop

model_norm.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_norm.add(BatchNormalization())#Normalizing the layer
model_norm.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')) 
model_norm.add(BatchNormalization())#Normalizing the layer
model_norm.add(MaxPooling2D((2, 2)))#Maxpool on the layer
model_norm.add(Dropout(0.4))#40% Value Drop

model_norm.add(Flatten()) #Flatten the data as array
model_norm.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_norm.add(BatchNormalization())
model_norm.add(Dropout(0.5)) #50% Value Drop
model_norm.add(Dense(10, activation='softmax'))

#data augmentation
datagen = ImageDataGenerator (width_shift_range=0.2,
                            height_shift_range=0.1,horizontal_flip=True,
                            vertical_flip=False) 
datagen.fit(trainX)
model_norm.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model_norm.fit_generator(datagen.flow(trainX, trainY, batch_size = 64),
                                 validation_data = (testX, testY),
                                 epochs = 25, verbose = 1)

_, acc = model_norm.evaluate(testX, testY, verbose=1)
print('> %.3f' % (acc * 100.0))


pyplot.title('Cross Entropy Loss') 
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')
pyplot.legend()
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')

# plot accuracy
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
pyplot.legend()
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')