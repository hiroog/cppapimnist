#!/usr/bin/python3
import urllib.request
import os
import gzip

DOWNLOAD_URL='http://yann.lecun.com/exdb/mnist/'
file_list=[ 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte' ]

for name in file_list:
    if not os.path.exists( name ):
        gz_name= name + '.gz'
        if not os.path.exists( gz_name ):
            print( 'download', gz_name )
            with urllib.request.urlopen( DOWNLOAD_URL + gz_name ) as fi:
                with open( gz_name, 'wb' ) as fo:
                    fo.write( fi.read() )
        print( 'write', name )
        with gzip.open( gz_name, 'rb' ) as fi:
            with open( name, 'wb' ) as fo:
                fo.write( fi.read() )



