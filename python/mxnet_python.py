# 2020/02/09 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import mxnet
from mxnet.gluon import nn as l
from mxnet import autograd
import numpy as np
import mnist_loader
import minitimer


def def_model():
    model= l.Sequential()
    model.add( l.Conv2D( 16, kernel_size=(5,5), activation='relu' ) )
    model.add( l.MaxPool2D( pool_size=(2,2) ) )
    model.add( l.Dropout( 0.25 ) )
    model.add( l.Conv2D( 32, kernel_size=(5,5), activation='relu' ) )
    model.add( l.MaxPool2D( pool_size=(2,2) ) )
    model.add( l.Dropout( 0.25 ) )
    model.add( l.Flatten() )
    model.add( l.Dense( 128, activation='relu' ) )
    model.add( l.Dropout( 0.5 ) )
    model.add( l.Dense( 64, activation='relu' ) )
    model.add( l.Dropout( 0.5 ) )
    model.add( l.Dense( 10 ) )
    return  model


def test_train():
    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    EPOCH=2
    BATCH_SIZE=32
    loop_count= len(x_train) // BATCH_SIZE

    device= mxnet.gpu() if mxnet.context.num_gpus() else mxnet.cpu()

    model= def_model()
    model.collect_params().initialize( mxnet.init.Xavier(), ctx=device )

    optimizer= mxnet.gluon.Trainer( model.collect_params(), 'adam', { 'learning_rate': 0.001 } )
    loss_func= mxnet.gluon.loss.L2Loss()

    with minitimer.Timer( 'train ' ):
        for ei in range(EPOCH):
            total_loss= 0.0
            for di in range(loop_count):
                rindex= np.random.randint( len(x_train), size=BATCH_SIZE )
                x_data= mxnet.nd.array( x_train[ rindex ] ).as_in_context( device )
                y_data= mxnet.nd.array( y_train[ rindex ] ).as_in_context( device )
                with autograd.record():
                    outputs= model( x_data )
                    loss= loss_func( outputs, y_data )
                    loss.backward()
                optimizer.step( x_data.shape[0] )
                total_loss+= loss[0].asscalar()
            print( ei, 'loss=', total_loss / loop_count )

    model.save_parameters( 'python_mnist_mxnet_python.params' )


def test_predict():
    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    BATCH_SIZE=64
    loop_count= len(x_test) // BATCH_SIZE

    device= mxnet.gpu() if mxnet.context.num_gpus() else mxnet.cpu()

    model= def_model()
    model.load_parameters( 'python_mnist_mxnet_python.params', ctx=device )

    with minitimer.Timer( 'predict ' ):
        score= 0
        for i in range(loop_count):
            rand_index= np.random.randint( len(x_test), size=BATCH_SIZE )
            x_data= mxnet.nd.array( x_test[rand_index] ).as_in_context( device )
            y_data= y_test[rand_index]
            result= model( x_data ).asnumpy()
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


