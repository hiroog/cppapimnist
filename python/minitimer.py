# 2020/02/09 Hiroyuki ogasawara
# vim:ts=4 sw=4 et:

import time

class Timer:
    def __init__( self, text= '' ):
    	self.text= text
    def __enter__( self ):
        self.start= time.perf_counter()
    def __exit__( self, type, value, trace ):
        dtime= time.perf_counter() - self.start
        print( self.text + 'time=', dtime )



