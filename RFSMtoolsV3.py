import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from datetime import timedelta,date,datetime
import time
import rioxarray
import rasterio
import xarray as xr
from osgeo import gdal, ogr, osr, os
from IPython.display import clear_output

from xrspatial import zonal_stats
import gzip
import urllib
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3
    
import cfgrib
from rasterstats import zonal_stats
import warnings
warnings.filterwarnings('ignore')
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class ncep():
    def __init__(self, workspace,date,dem, source='NCEP/PecipRate'):
        self.workdir = workspace
        self.path = os.path.join(self.workdir,'RadarPrecip')
        if not os.path.isdir(self.path): 
            os.mkdir(self.path)
        os.chdir(self.path)
        self.date = date
        self.source = source
        tif = xr.open_dataset(dem)
        self.crs = tif.rio.crs.wkt
        self.min_x = min(tif.x.data)
        self.min_y = min(tif.y.data)
        self.max_x = max(tif.x.data)
        self.max_y = max(tif.y.data)
        self.timestep = 20
    
    def get_url(self):
        date=self.date
        if self.source == 'NCEP/PecipRate':
            self.dt = timedelta(minutes=self.timestep)
            url_ = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/PrecipRate/PrecipRate_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(
                date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        if self.source == 'NCEP/QPE01hH':
            self.dt = timedelta(hours=1)
            url_ = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/GaugeCorr_QPE_01H/GaugeCorr_QPE_01H_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(
                date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        return url_
    
    def dl(self):
        date=self.date
        url = self.get_url()
        filename = url.split("/")[-1][:-3]
        if not os.path.isfile(filename):
            print('dl NEXRAD')
            with urllib.request.urlopen(url) as response:
                with gzip.GzipFile(fileobj=response) as uncompressed:
                    file_content = uncompressed.read()
            with open(filename, 'wb') as f:
                f.write(file_content)
            
    def clip(self):
        filename=self.get_url().split("/")[-1][:-3]
        ingrib = os.path.join(self.path,filename)
        ds = xr.open_dataset(ingrib, engine="cfgrib", decode_coords="all")

        #precipitation radar dataset misses the correct ESPG factory code
        xds = ds.rio.write_crs(4326)
        xds = xds.rio.reproject(self.crs) #this lines takes few seconds
        subset_xds= xds.rio.clip_box(minx=self.min_x,miny=self.min_y,maxx=self.max_x,maxy=self.max_y)
        if sum(sum(subset_xds.unknown.data))>0:
            # Un/Comment to store rainfall
#             name = str(date.year) + str(date.month)+str(date.day)+str(date.hour)+".tif"
#             subset_xds.rio.to_raster(name)
            rain = np.average(subset_xds.unknown.data)

            rain = rain*(self.timestep/60) #the unit is mm/hour and the dt is 4 minutes

        else:
            rain = 0
        return rain     

class graph():
    def __init__ (self, tables, workspace, DOI=None): 
        self.net = pd.read_excel(tables, sheet_name ='sheet_Graph_df')
        self.basins = pd.read_excel(tables, sheet_name ='sheet_Subbasins')
        self.basins['child'] = self.basins['Node'].apply(lambda x: x.replace('subbasin', 'L1'))
        if sum(self.basins['CN'].isna())>0 :
            self.basins['CN'] = self.basins['CN'].fillna(84)
            print('missing CN values are set to 84')
        self.source = workspace
        self.DOI = DOI

    def sub_graph(self,DG, N):
        up = [n for n in nx.traversal.bfs_tree(DG, N, reverse=True)]
        down = [n for n in nx.traversal.bfs_tree(DG, N, reverse=False) if (n != N)]
        self.NDOI = up + down
        subDG = nx.DiGraph((u, v, e) for u,v,e in DG.edges(data=True) if u in self.NDOI)
        nx.set_node_attributes(subDG, dict([(u,v) for u,v in zip(self.net.Node, self.net.surface) if u in self.NDOI]), name='surface')
        return subDG
    
    def dry_net(self): 
        DG = nx.DiGraph()
        weights = list(zip(self.net.Node.values,self.net.child.values, self.net['volume-m3'].values))
        DG.add_weighted_edges_from(weights)
        weights_subbasins = list(zip(self.basins.Node.values,self.basins.child.values,[0]*len(self.basins)))
        nx.set_node_attributes(DG, dict(zip(self.net.Node, self.net.surface)), name="surface")
        DG.add_weighted_edges_from(weights_subbasins)
        if self.DOI:
            DG = self.sub_graph(DG.copy(),self.DOI)
        self.DG = DG
        return DG
    
    def get_Q(self, acc_rain,CN):
        S = (25400/CN) - 254
        Ia = 0.2*S
        if acc_rain<=Ia: 
            Q=0
        if acc_rain>Ia:
            Q=((acc_rain-Ia)**2)/(acc_rain+(0.8*S))
        return Q

    def get_rain_edges(self, rain, acc_rain):
        if self.DOI: 
            basins = self.basins[self.basins.child.isin(self.NDOI)]
        else: 
            basins=self.basins
        basins['Q'] = basins['CN'].apply(lambda x: self.get_Q(acc_rain, x))
        basins['R'] = basins['Q']/acc_rain
        basins['runoff'] = basins.apply(lambda x: rain*x.R*x.AREA*0.001, axis=1) #(cubic meter)
        R = ['R']*len(basins)
        rain_edges = list(zip(R,basins.Node,basins.runoff))
        self.rain_edges = rain_edges
        return (rain_edges)
        
    def control_merge(self, DG):
        '''if any merged filling in nodes (L2>)'''
        nodes = list(DG.successors('R'))
        merges =[ m for m in [n for n in nodes if n.startswith('L')] if int(m.split("-")[0].split('L')[1])>1]
        M=len(merges)
        sp = dict(nx.all_pairs_shortest_path(DG))
        while M>0:
            temp=[]
            for node in merges:
                surface = DG.nodes[node]['surface']

                #all the predecessors of a node if there surface is lower than the node
                up = [n for n in nx.traversal.bfs_tree(DG, node, reverse=True)if (n != node) &(n.startswith('L'))]
                up = [n for n in up if DG.nodes[n]['surface']<surface]

                # predecessors that all nodes in their path to node have lower surface
                up_ = [n for n in up if
                    (max([DG.nodes[j]['surface'] for j in sp[n][node] if j!=node])<surface)
                      ]

                # predecessors that have more than 0 capacity 
                up__= [n for n in up_ if
                      DG[n][list(DG.successors(n))[0]]['weight']<0
                      ]
                surfaces = [DG.nodes[n]['surface'] for n in up__]
                sorted_up = [x for _,x in sorted(zip(surfaces,up__)) ]
                cap = [DG[n][list(DG.successors(n))[0]]['weight'] for n in sorted_up]

                L=int(node.split("-")[0].split('L')[1])
                prereq = sorted_up
                temp.append(len(prereq))
                if len(prereq)>0: 
                    WRn = DG['R'][node]['weight']
                    DG.remove_edge('R',node)
                    for n in prereq:
                        # Move rain from node to n
                        cap = abs(DG[n][list(DG.successors(n))[0]]['weight'])
                        
                        if DG.has_edge('R',n):
                            cap_ = max(cap-DG['R'][n]['weight'], 0)
                            Wmerge_prereq = min(cap_,WRn)
                            Wmerge_prereq = Wmerge_prereq + DG['R'][n]['weight']
                            DG.remove_edge('R',n)
                            DG.add_weighted_edges_from([('R',n,Wmerge_prereq)])
                        else:
                            Wmerge_prereq = min(cap,WRn)
                            DG.add_weighted_edges_from([('R',n,Wmerge_prereq)])
                        WRn = WRn - Wmerge_prereq
                        #print('The runoff volume of {} is added to {}'.format(WRn,n))
                    if WRn>0: 
                        DG.add_weighted_edges_from([('R',node,WRn)])
                        #print('No prereg could handle merge of {} overflow remains {}'.format(node, WRn))
                    
            M=0
            #if any(temp):M = max(temp)
            nodes = list(DG.successors('R'))
            merges =[ m for m in [n for n in nodes if n.startswith('L')] if int(m.split("-")[0].split('L')[1])>1]
        return DG.copy()   
        
    def fill_spill(self, wetDG):
        nodes = list(wetDG.successors('R'))
        updateG = wetDG.copy()
        updateG.remove_node('R')
        spill = 0
        while len(nodes)>0: 
            '''update the network to prevent bubble merging'''
            '''find edges that connect R to merged depressions in the WetDG'''
            wetDG = self.control_merge(wetDG)
            nodes = list(wetDG.successors('R'))
            spill+= 1
            for n in nodes: 
                if n == 'outfall': continue
                child_n = list(wetDG.successors(n))[0]
                WRn = wetDG['R'][n]['weight']
                WnCn = wetDG[n][child_n]['weight']  
                un_WRCn = max(0, WRn + WnCn) #spill from n to n-child

                un_WnCn = min(0, WRn + WnCn) #reduced capacity of n if it is partially filled (no-spill)  
                updateG.add_weighted_edges_from([(n,child_n,un_WnCn)])
                if updateG.has_edge('R',child_n):
                    new_un_WRCn = un_WRCn + updateG['R'][child_n]['weight']
                    un_WRCn = new_un_WRCn
                    updateG.remove_edge('R',child_n)#new line#
                if un_WRCn> 0 : 
                    updateG.add_weighted_edges_from([('R',child_n,un_WRCn)])
                    
                #### merge correction to acount for prereqities depressions before merging
            if updateG.has_node('R') == False: 
                nodes = []
                #print('end of spilling')
            if updateG.has_node('R') == True: 
#                 print('has node R')
                wetDG = updateG.copy()
                updateG.remove_node('R')
                nodes = list(wetDG.successors('R'))
                #print(nodes)
        return updateG



class post_process():
    def __init__(self, simulation, tables,ncfile, road_raster=None):
        self.sim = simulation #simulation is a dict
        self.wnet = pd.DataFrame(self.sim)
        self.tnet = pd.read_excel(tables, sheet_name ='sheet_Graph_df')
        self.dnet = self.tnet[(self.tnet.Node.isin(self.wnet.Node)) & (self.tnet.Node.str.startswith('L'))]
        self.ncdf = xr.open_dataset(ncfile)
        vol = pd.read_excel(tables, sheet_name ='sheet_Volume2Depth')
        vol = vol.replace(0, np.nan)
        vol['step0'] = 0
        vol.index = vol.Node
        vol = vol.drop(columns = ["VALUE", "Node","Unnamed: 0"], errors='ignore')
        self.vol = vol.T
        self.rr = np.ndarray(self.ncdf['watershed'][:,:,1].data.shape)
        if road_raster != None:
            self.rr = rasterio.open(road_raster).read_masks(1)

    def get_inundation_step(self):
        net = self.dnet[self.dnet.Node.str.startswith('L')][['Node', 'volume-m3']]
        fnet = pd.merge(net,self.wnet, on='Node')
        fnet['filled'] = abs(fnet['volume-m3']) - abs(fnet['remained-volume-m3'])
        fnet ['step'] = fnet.apply(lambda row: self.get_nearest_step(row.Node, row.filled), axis=1)
        self.fnet = fnet
        return fnet
    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def get_nearest_step(self, node, value):
        try:
            a = self.vol[node][self.vol[node].notna()]
            return int(float(a.index[self.find_nearest(value, a)][4:]))
        except KeyError:
            return 0
        
    
    def get_net(self, netdf, edge_feature): 
        DG = nx.DiGraph()
        weights = list(zip(netdf.Node.values,netdf.child.values, netdf[edge_feature].values))
        DG.add_weighted_edges_from(weights)
        return(DG)
        
    
    def plot_net(self,net,name):
        plt.figure(figsize=(30,20))
        pos = graphviz_layout(net, prog="sfdp")
        nx.draw(net,pos=pos,with_labels=True,node_size=50, font_size = 40,arrowsize=40, arrowstyle='simple')
        labels = nx.get_edge_attributes(net,'weight')
        labels2 = {}
        for k,v in labels.items():
            labels2[k] = format(v*-1,".2f")
        nx.draw_networkx_edge_labels(net,pos,edge_labels=labels2)
        plt.savefig(r'C:\Research\Notebooks\Clowder RFSM\{}'.format(name),dpi=300)
        plt.show()
    def to_depth(self,i): 
        return self.mapping[int(i)]
    def net_to_tif(self, fnet, tables):
        ncdf = self.ncdf
        inundation = np.ndarray(ncdf['watershed'][:,:,1].data.shape)

        for l in [1,2,3,4,5,6,7]:
            surface = np.nan_to_num(ncdf['surface'][:,:,l].data,0)
            bottom = np.nan_to_num(ncdf['bottom'][:,:,l].data,0)
            watershed_ar = ncdf['watershed'][:,:,l].data
            if np.amax(ncdf['watershed'][:,:,l].data)>0:
                tnet = self.tnet[self.tnet.Node.str.startswith('L'+str(l))]
                tnet['watershed'] = tnet.Node.apply(lambda x: int(x[3:]))
                wsh = np.unique(watershed_ar).astype(int)
                if self.rr.any():
                    rrwsh = np.unique(watershed_ar[np.where(self.rr!= 0)]).astype(int)
                    names = ['L'+str(int(l))+'-'+str(int(x)) for x in rrwsh if x>0]
                rest =  np.array(tnet['watershed'])   
                fnet_l = fnet[fnet.Node.isin(names)][['Node','step']]
                fnet_l['watershed'] = fnet_l['Node'].apply(lambda x: int(x[3:]))
                fnet_l['depth'] = fnet_l['step']*0.1524#step_size
                
                source = fnet_l['watershed'].values
                out = fnet_l['depth'].values
                rest_ = np.array((list(set(rest)-set(source))))
                mapping1 = {source[a]:out[a] for a in range(len(fnet_l))}
                mapping2 = {rest_[a]:0 for a in range(len(rest_))}
                mapping3 = {0:0}
                self.mapping = {**mapping1, **mapping2, **mapping3}
                to_depthv = np.vectorize(self.to_depth, otypes=[float])
                inshape=watershed_ar.shape
                depth = np.nan_to_num(to_depthv(watershed_ar),0)
                top = np.minimum(bottom+depth,surface)
                thresh = np.amax(depth)
                inund = np.clip(top - np.nan_to_num(ncdf['dem'][:,:,l].data,0), a_min=0, a_max=100)
                inundation = inundation+inund
        
        return inundation
        
    def array_to_tif(rasterfn,newRasterfn,array):
        raster = gdal.Open(rasterfn)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        cols = raster.RasterXSize
        rows = raster.RasterYSize

        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()

class manual_source(graph):
    def __init__(self, tables, workspace, source_node, overflow):
        self.G = graph(tables, workspace)
        R = ['R']*len(source_node)
        self.rain_edges = list(zip(R,source_node,overflow))
        print(self.rain_edges)
    def run(self):
        DG = self.G.dry_net()
        DG.add_weighted_edges_from(self.rain_edges)
        self.newDG = self.G.fill_spill(DG)
        df = nx.to_pandas_edgelist(self.newDG).rename(columns={'source':'Node', 'target':'child', 'weight':'remained-volume-m3'})
        return df.to_dict()
        
        
class RFSM(graph,ncep, post_process):
    def __init__(self,start, end, dem, tables, workspace, ncdf,road_raster,DOI=None, rain_source='NCEP/PecipRate'):
        self.time_format = "%Y-%m-%d %H:%M:%S"
        self.start = datetime.strptime(start, self.time_format)
        self.end=datetime.strptime(end, self.time_format)
        self.dem=dem
        self.workspace=workspace
        self.net = pd.read_excel(tables, sheet_name ='sheet_Graph_df')
        self.tables=tables
        self.ncdf = ncdf
        self.road_raster = road_raster
        self.rr = np.nan_to_num(xr.open_dataset(road_raster).band_data.data[0],0)
        self.segs = list(np.unique(self.rr).astype(int))
        self.segs.remove(0)
        self.rain_source = rain_source
        self.DOI = DOI
    def getsegment(self,inund,t,seg):
        max_depth = max((inund[np.where(self.rr==seg)]))
        area = 9*len(inund[np.where((self.rr==seg)&(inund>0))])
        return [seg, t, max_depth, area]
    def run(self):
        acc_rain = 0
        t=self.start
        self.G = graph(self.tables, self.workspace, self.DOI)
        DG = self.G.dry_net()
        i=1
        simulation = dict()
        rrinund=pd.DataFrame()
        last_depths = 0
        thresh=0
        self.ncep_rain = pd.DataFrame()
        while t<self.end:
            self.ncep = ncep(self.workspace, t, self.dem, self.rain_source )
            try: 
                self.ncep.dl()
                rain = self.ncep.clip()
            except: 
                rain = rain
                print('failed NEXRAD dl')
            self.ncep_rain = self.ncep_rain.append([[rain,t]])
            print(t,'-----> accumulated rain:{}----instantaneous rain:{}'.format(acc_rain,rain))
            acc_rain += rain
            rain_edges = self.G.get_rain_edges(rain, acc_rain)#based on cn gets the runoff to each subbasin
            DG.add_weighted_edges_from(rain_edges)
            newDG = self.G.fill_spill(DG)
            

            DG = newDG.copy()
            df = nx.to_pandas_edgelist(DG).rename(columns={'source':'Node', 'target':'child', 'weight':'remained-volume-m3'})
            simulation[t] = df.to_dict()
            t = self.ncep.date + self.ncep.dt
            i += 1
            pp = post_process(df.to_dict(),self.tables, self.ncdf,self.road_raster)
            wetdf = pp.get_inundation_step()
            depths = wetdf.step.values 
            thresh = max(depths-last_depths) 
            if thresh>0: 
                inund = pp.net_to_tif(wetdf,self.tables)
                rrinund = rrinund.append([self.getsegment(inund,t,seg) for seg in self.segs])
                last_depths = depths
        self.simulation = simulation
        self.road_inundation = rrinund.rename(columns={0:'segment',1:'time',2:'max_depth',3:'area'})
        return simulation, self.road_inundation
