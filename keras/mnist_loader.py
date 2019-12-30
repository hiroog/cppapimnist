# 2019/12/30 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import os
import struct
import numpy as np

class MNistLoader:

    def __init__( self, data_path ):
        self.mnist_data= []
        self.load( data_path )

    def load( self, data_path ):
        mnist_files= [
            'train-images-idx3-ubyte',
            't10k-images-idx3-ubyte',
            'train-labels-idx1-ubyte',
            't10k-labels-idx1-ubyte',
        ];
        for name in mnist_files:
            full_path= os.path.join( data_path, name )
            with open( full_path, 'rb' ) as fi:
                self.mnist_data.append( fi.read() )

    def byteToFloat( self, data ):
        float_data= np.zeros( 28*28, dtype='float' )
        for i,b in enumerate(data):
            float_data[i]= b/255.0
        return  float_data

    def getImage( self, index, data_index ):
        image= self.mnist_data[data_index]
        base= 16+28*28*index
        return  self.byteToFloat( image[base:base+28*28] )

    def getFloatImage( self, index ):
        return  self.getImage( index, 0 )

    def getFloatImageTest( self, index ):
        return  self.getImage( index, 1 )

    def getLabel( self, index ):
        label= self.mnist_data[2]
        return  int(label[8 + index])

    def getLabelTest( self, index ):
        label= self.mnist_data[3]
        return  int(label[8 + index])

    def getFloatLabel10( self, index ):
        label= np.zeros( 10, dtype='float' )
        label[self.getLabel(index)]= 1.0
        return  label

    def getFloatLabel10Test( self, index ):
        label= np.zeros( 10, dtype='float' )
        label[self.getLabelTest(index)]= 1.0
        return  label

    def getAll( self ):
        x_train= np.ndarray( shape=(60000,1,28,28), dtype='float' )
        y_train= np.ndarray( shape=(60000,10), dtype='float' )
        for i in range(60000):
            x_train[i]= self.getFloatImage( i ).reshape(1,28,28)
            y_train[i]= self.getFloatLabel10( i )
        x_test= np.ndarray( shape=(10000,1,28,28), dtype='float' )
        y_test= np.ndarray( shape=(10000,10), dtype='float' )
        for i in range(10000):
            x_test[i]= self.getFloatImageTest( i ).reshape(1,28,28)
            y_test[i]= self.getFloatLabel10Test( i )
        return  (x_train,y_train),(x_test,y_test)


def print28( data ):
    for y in data:
        a= ''
        for x in y:
            if x > 0.5:
                a+='#'
            else:
                a+='.'
        print( a )


def main():
    loader= MNistLoader( '../mnist' )
    for i in range(10):
        index= i
        print( loader.getLabel( index ) )
        print( loader.getFloatLabel10( index ) )
        print28( loader.getFloatImage( index ).reshape(28,28) )


if __name__=='__main__':
    main()




