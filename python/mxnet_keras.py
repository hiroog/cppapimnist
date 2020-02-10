# 2020/02/09 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import os
os.environ['KERAS_BACKEND']='mxnet'

import keras
from keras import models
from keras import layers as l
import numpy as np
import mnist_loader
import minitimer


# channels_first + functional API + mse

def test_train():
    x0= l.Input( shape=(1,28,28) )
    x= l.Conv2D( 16, (5,5), kernel_initializer='he_normal', activation='relu' )( x0 )
    x= l.MaxPooling2D( pool_size=(2,2) )( x )
    x= l.Dropout( 0.25 )( x )
    x= l.Conv2D( 32, (5,5), kernel_initializer='he_normal', activation='relu' )( x )
    x= l.MaxPooling2D( pool_size=(2,2) )( x )
    x= l.Dropout( 0.25 )( x )
    x= l.Flatten()( x )
    x= l.Dense( 128, kernel_initializer='he_normal', activation='relu' )( x )
    x= l.Dropout( 0.5 )( x )
    x= l.Dense( 64, kernel_initializer='he_normal', activation='relu' )( x )
    x= l.Dropout( 0.5 )( x )
    x= l.Dense( 10, kernel_initializer='he_normal' )( x )
    model= models.Model( inputs=x0, outputs=x )
    model.compile( loss='mse', metrics=['accuracy'], optimizer=keras.optimizers.Adam( lr=0.01 ) )
    model.summary()

    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    with minitimer.Timer( 'train ' ):
        model.fit( x_train, y_train, batch_size=32, epochs=2, verbose=1 )
    model.save( 'python_mnist_mxnet_keras.h5' )


def test_predict():
    BATCH_SIZE=64
    model= models.load_model( 'python_mnist_mxnet_keras.h5' )

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


