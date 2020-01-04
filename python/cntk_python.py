# 2019/12/30 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import os
os.environ['KERAS_BACKEND']='cntk'

import cntk
from cntk import layers as l
import numpy as np
import mnist_loader


# channels_first + sequential API + softmaxcrossentropy

def test_train():
    xinput= cntk.input_variable( (1,28,28), np.float32, name='xinput' )
    yinput= cntk.input_variable( (10), np.float32 )
    model= l.Sequential( [
            l.Convolution( (5,5), 16, pad=False, activation=cntk.relu ),
            l.MaxPooling( (2,2), strides=2 ),
            l.Dropout( 0.25 ),
            l.Convolution( (5,5), 16, pad=False, activation=cntk.relu ),
            l.MaxPooling( (2,2), strides=2 ),
            l.Dropout( 0.25 ),
            l.Dense( 128, activation=cntk.relu ),
            l.Dropout( 0.5 ),
            l.Dense( 64, activation=cntk.relu ),
            l.Dropout( 0.5 ),
            l.Dense( 10, activation=None )
        ] )
    youtput= model( xinput )
    loss= cntk.cross_entropy_with_softmax( youtput, yinput )
    cntk.debugging.dump_function( youtput )
    optimizer= cntk.adam( youtput.parameters, cntk.learning_rate_schedule( 0.01, cntk.UnitType.minibatch ), 0.9 )
    trainer= cntk.Trainer( youtput, loss, [optimizer], [cntk.logging.ProgressPrinter(0)] )

    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    EPOCH=2
    BATCH_SIZE=32
    loop_count= len(x_test) // BATCH_SIZE
    for e in range(EPOCH):
        for i in range(loop_count):
            rindex= np.random.randint( len(x_test), size=BATCH_SIZE )
            x_data= x_train[ rindex ]
            y_data= y_train[ rindex ]
            trainer.train_minibatch( { xinput: x_data, yinput: y_data } )

    youtput.save( 'python_mnist_cntk_python.dnn' )


def test_predict():
    BATCH_SIZE=64
    model= cntk.ops.functions.load_model( 'python_mnist_cntk_python.dnn' )

    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    loop_count= len(x_test) // BATCH_SIZE

    score= 0
    for i in range(loop_count):
        rand_index= np.random.randint( len(x_test), size=BATCH_SIZE )
        x_data= x_test[rand_index]
        y_data= y_test[rand_index]
        result= model.eval( { 'xinput': x_data } )
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


