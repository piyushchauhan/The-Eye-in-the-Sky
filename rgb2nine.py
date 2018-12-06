import numpy as np

Water = [0,0,150]
Trees = [0,125,0] 
Grass = [0,255,0] 
Trains = [255,255,0] 
Soil = [150,80,0] 
Roads = [0,0,0] 
Unlabelled = [255,255,255] 
Buildings = [100,100,100] 
Pools = [150,150,255]

def testto9(image):
    wctmp = (image == Water)
    wc = np.logical_and(wctmp[:,:,0], wctmp[:,:,1]) 
    wc = np.logical_and(wctmp[:,:,2], wc)

    tretmp = (image == Trees)
    tre = np.logical_and(tretmp[:,:,0], tretmp[:,:,1]) 
    tre = np.logical_and(tretmp[:,:,2], tre)

    gtmp = (image == Grass)
    g = np.logical_and(gtmp[:,:,0], gtmp[:,:,1]) 
    g = np.logical_and(tretmp[:,:,2], g)

    tratmp = (image == Trains)
    tra = np.logical_and(tratmp[:,:,0], tratmp[:,:,1]) 
    tra = np.logical_and(tratmp[:,:,2], tra)

    stmp = (image == Soil)
    s = np.logical_and(stmp[:,:,0], stmp[:,:,1]) 
    s = np.logical_and(stmp[:,:,2], s)

    rtmp = (image == Roads)
    r = np.logical_and(rtmp[:,:,0], rtmp[:,:,1]) 
    r = np.logical_and(rtmp[:,:,2], r)

    btmp = (image == Buildings)
    b = np.logical_and(btmp[:,:,0], btmp[:,:,1]) 
    b = np.logical_and(btmp[:,:,2], b)

    ptmp = (image == Pools)
    p = np.logical_and(ptmp[:,:,0], ptmp[:,:,1]) 
    p = np.logical_and(ptmp[:,:,2], p)

    utmp = (image == Unlabelled)
    u = np.logical_and(utmp[:,:,0], utmp[:,:,1])
    u = np.logical_and(utmp[:,:,2], u)

    output = np.zeros((image.shape[0],image.shape[1],9))
    output[:,:,0] = wc #Water Channel
    output[:,:,1] = tre #Trees
    output[:,:,2] = g #Grass
    output[:,:,3] = tra #Trains
    output[:,:,4] = s #Soil 
    output[:,:,5] = r #Roads
    output[:,:,6] = b #Buildings
    output[:,:,7] = p #Pools
    output[:,:,8] = u #Unlabelled
