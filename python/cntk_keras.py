# 2019/12/30 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import os
os.environ['KERAS_BACKEND']='cntk'

import keras
from keras import models
from keras import layers as l
import numpy as np
import mnist_loader
import minitimer


# channels_first + sequential API + softmaxcrossentropy

def test_train():
    model= models.Sequential()
    model.add( l.Conv2D( 16, kernel_size=(5,5), activation='relu', input_shape=(1,28,28) ) )
    model.add( l.MaxPooling2D( pool_size=(2,2) ) )
    model.add( l.Dropout( 0.25 ) )
    model.add( l.Conv2D( 32, kernel_size=(5,5), activation='relu' ) )
    model.add( l.MaxPooling2D( pool_size=(2,2) ) )
    model.add( l.Dropout( 0.25 ) )
    model.add( l.Flatten() )
    model.add( l.Dense( 128, activation='relu' ) )
    model.add( l.Dropout( 0.5 ) )
    model.add( l.Dense( 64, activation='relu' ) )
    model.add( l.Dropout( 0.5 ) )
    model.add( l.Dense( 10, activation='softmax' ) )
    model.compile( loss= 'categorical_crossentropy', metrics=['accuracy'], optimizer='adam' )
    model.summary()

    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    with minitimer.Timer( 'train ' ):
        model.fit( x_train, y_train, batch_size=32, epochs=2, verbose=1 )
    model.save( 'python_mnist_cntk_keras.h5' )


def test_predict():
    BATCH_SIZE=64
    model= models.load_model( 'python_mnist_cntk_keras.h5' )

    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    loop_count= len(x_test) // BATCH_SIZE

    with minitimer.Timer( 'predict ' ):
        score= 0
        for i in range(loop_count):
            rand_index= np.random.randint( len(x_test), size=BATCH_SIZE )
            x_data= x_test[rand_index]
            y_data= y_test[rand_index]
            result= model.predict( x_data, batch_size=BATCH_SIZE, verbose=0 )
            for ba,bb in zip(result,y_data):
                ra= np.argmax( ba )
                rb= np.argmax( bb )
                if ra == rb:
                    score+= 1
    print( score * 100.0 / (loop_count * BATCH_SIZE), '%' )



def main():
    test_train()
    test_predict()

if __name__=='__main__':
    main()


