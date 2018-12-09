road = 15.2
building = 21.88
tree = 8.5  
grass = 6.88
soil = 0.93
water = 5.72
rail = 1.0
pool = 0.15
unlabelled = 39.72
temp = [water, tree, grass, rail, soil, road, building, pool, unlabelled]

def perc():
    return [x / 100 for x in temp] 