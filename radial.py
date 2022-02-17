import termplotlib as tpl
import requests,json,random,time,shutil
from shapely.geometry import Polygon, Point, shape
from shapely.ops import cascaded_union
from p_tqdm import p_map
import tqdm as tqdm
import numpy as np
# import pandas as pd

__domain__ = 'http://ec2-18-193-78-190.eu-central-1.compute.amazonaws.com:25252/'
__area__ = 'geotype=LSOA'
__item__ = 'cols=geography_code,KS102EW0001'
__query__ = 'query/2011'

n_samples= 1e4


# get a polygon somewhat representative of England
__england__ = json.load(open('./Countries_(December_2017)_Boundaries.geojson'))['features'][0]['geometry']
shapes = [x.buffer(0) for x in shape(__england__).buffer(0).geoms]
shapes.sort(key=lambda x: x.area, reverse=True)
poly = cascaded_union(shapes[:10])



query = 'health'
res = requests.get(f'{__domain__}{query}')
rtn = json.dumps(res.json(),indent=4)
# print(rtn)



def randomInPoly(_):
     minx, miny, maxx, maxy = poly.bounds
     while True:
         p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         if poly.contains(p):
             xy = p.coords.xy
             return '%.04d'%xy[0][0], '%.04d'%xy[1][0]


sample = p_map( randomInPoly, range(int(n_samples)) )


def get(c):
    # c = [0.1338,51.4635]
    '''
    Make a query String
    '''
    dist = int(random.random()*10000) # 5km radius 
    rad = f'location={c[0]},{c[1]}&radius={dist}'
    qst = f'{__domain__}{__query__}?{rad}&{__item__}'
    # print(qst)

    '''
    Send request
    '''
    s = time.time_ns()
    res = requests.get(qst)
    e = time.time_ns()

    duration = (e-s)/1e9
    nres= res.text.count('\n')-1

    # print(res.text,nres,duration)
    return nres,duration


# r = get(sample[7])

# outputs = [get(c) for c in tqdm.tqdm(sample)]
outputs  = p_map(get, sample)

term = shutil.get_terminal_size((80, 20))
fig = tpl.figure() 
x = [i[0] for i in outputs]
y = [i[1] for i in outputs]
fig.plot(x, y, label="# vs duration", width=term.columns, height=term.lines)
fig.show()

print('\nFrequency')

counts, bin_edges = np.histogram(y)
fig = tpl.figure() 
fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False) 
fig.show()


df = pd.DataFrame(outputs,columns=['#out','duration'])
df.to_csv('radial_outputs1.csv')

import matplotlib.pyplot as plt
plt.scatter(x,y,s=1,alpha=.4)
plt.xlabel('Areas Found')
plt.ylabel('Call Duration (s)')
plt.savefig('radial_outputs1.png')