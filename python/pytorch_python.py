# 2019/12/30 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mnist_loader


class Model_MNist( nn.Module ):

    def __init__( self ):
        super().__init__()
        self.c0= nn.Conv2d( 1, 16, 5 )
        self.c1= nn.Conv2d( 16, 32, 5 )
        self.fc0= nn.Linear( 32*4*4, 128 )
        self.fc1= nn.Linear( 128, 64 )
        self.fc2= nn.Linear( 64, 10 )
        self.drop0= nn.Dropout( 0.25 )
        self.drop1= nn.Dropout( 0.25 )
        self.drop2= nn.Dropout( 0.5 )
        self.drop3= nn.Dropout( 0.5 )

    def forward( self, x ):
        x= F.relu( self.c0( x ) )
        x= F.max_pool2d( x, (2, 2) )
        x= self.drop0( x )
        x= F.relu( self.c1( x ) )
        x= F.max_pool2d( x, (2, 2) )
        x= self.drop1( x )
        x= x.view( x.size(0), -1 )
        x= F.relu( self.fc0( x ) )
        x= self.drop2( x )
        x= F.relu( self.fc1( x ) )
        x= self.drop3( x )
        x= self.fc2( x )
        return  x


def test_train():
    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    EPOCH=2
    BATCH_SIZE=32
    loop_count= len(x_train) // BATCH_SIZE

    device= torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    model= Model_MNist()
    model.to( device )
    optimizer= torch.optim.Adam( model.parameters(), lr=0.001 )
    loss_func= nn.MSELoss()

    model.train()
    for ei in range(EPOCH):
        total_loss= 0.0
        for di in range(loop_count):
            rindex= np.random.randint( len(x_train), size=BATCH_SIZE )
            x_data= torch.tensor( x_train[ rindex ], dtype=torch.float32, device=device )
            y_data= torch.tensor( y_train[ rindex ], dtype=torch.float32, device=device )

            optimizer.zero_grad()
            outputs= model( x_data )
            loss= loss_func( outputs, y_data )
            loss.backward()
            optimizer.step()
            total_loss+= loss.item()
        print( ei, 'loss=', total_loss / loop_count )

    torch.save( model.state_dict(), 'python_mnist_pytorch_python.pt' )


def test_predict():
    loader= mnist_loader.MNistLoader( '../mnist' )
    (x_train,y_train),(x_test,y_test)= loader.getAll()

    BATCH_SIZE=64
    loop_count= len(x_test) // BATCH_SIZE

    device= torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    model= Model_MNist()
    model.to( device )

    model.load_state_dict( torch.load( 'python_mnist_pytorch_python.pt' ) )

    model.eval()
    score= 0
    for di in range(loop_count):
        rand_index= np.random.randint( len(x_test), size=BATCH_SIZE )
        x_data= torch.tensor( x_test[rand_index], dtype=torch.float32, device=device )
        y_data= y_test[rand_index]
        outputs= model( x_data )
        outputs_c= outputs.to( 'cpu' ).detach().numpy()
        for ba,bb in zip(outputs_c,y_data):
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


