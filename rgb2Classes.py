import numpy as np

Water = np.array([0, 0, 150])
Trees = np.array([0, 125, 0])
Grass = np.array([0, 255, 0])
Trains = np.array([255, 255, 0])
Soil = np.array([150, 80, 0])
Roads = np.array([0, 0, 0])
Unlabelled = np.array([255, 255, 255])
Buildings = np.array([100, 100, 100])
Pools = np.array([150, 150, 255])


def rgb2Classes(image):
    wctmp = (image == Water)
    wc = np.logical_and(wctmp[:, :, 0], wctmp[:, :, 1])
    wc = np.logical_and(wctmp[:, :, 2], wc)
    wc = np.array(wc, dtype=int)


    tretmp = (image == Trees)
    tre = np.logical_and(tretmp[:, :, 0], tretmp[:, :, 1])
    tre = np.logical_and(tretmp[:, :, 2], tre)
    tre = np.array(tre, dtype=int)

    gtmp = (image == Grass)
    g = np.logical_and(gtmp[:, :, 0], gtmp[:, :, 1])
    g = np.logical_and(tretmp[:, :, 2], g)
    g = np.array(g, dtype=int)

    tratmp = (image == Trains)
    tra = np.logical_and(tratmp[:, :, 0], tratmp[:, :, 1])
    tra = np.logical_and(tratmp[:, :, 2], tra)
    tra = np.array(tra, dtype=int)

    stmp = (image == Soil)
    s = np.logical_and(stmp[:, :, 0], stmp[:, :, 1])
    s = np.logical_and(stmp[:, :, 2], s)
    s = np.array(s, dtype=int)

    rtmp = (image == Roads)
    r = np.logical_and(rtmp[:, :, 0], rtmp[:, :, 1])
    r = np.logical_and(rtmp[:, :, 2], r)
    r = np.array(r, dtype=int)

    btmp = (image == Buildings)
    b = np.logical_and(btmp[:, :, 0], btmp[:, :, 1])
    b = np.logical_and(btmp[:, :, 2], b)
    b = np.array(b, dtype=int)

    ptmp = (image == Pools)
    p = np.logical_and(ptmp[:, :, 0], ptmp[:, :, 1])
    p = np.logical_and(ptmp[:, :, 2], p)
    p = np.array(p, dtype=int)

    utmp = (image == Unlabelled)
    u = np.logical_and(utmp[:, :, 0], utmp[:, :, 1])
    u = np.logical_and(utmp[:, :, 2], u)
    u = np.array(u, dtype=int)

    output = np.zeros((image.shape[0], image.shape[1], 9))
    output[:, :, 0] = wc  # Water Channel
    output[:, :, 1] = tre  # Trees
    output[:, :, 2] = g  # Grass
    output[:, :, 3] = tra  # Trains
    output[:, :, 4] = s  # Soil
    output[:, :, 5] = r  # Roads
    output[:, :, 6] = b  # Buildings
    output[:, :, 7] = p  # Pools
    output[:, :, 8] = u  # Unlabelled
    return output
