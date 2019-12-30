#!/bin/sh

DOWNLOAD_URL=http://yann.lecun.com/exdb/mnist/

FILE_LIST="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"

for name in $FILE_LIST; do
    if [ ! -e $name ]; then
        gz_name=${name}.gz
        if [ ! -e $gz_name ]; then
            wget $DOWNLOAD_URL/$gz_name
        fi
        gzip -d $gz_name
    fi
done

