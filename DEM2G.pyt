import numpy as np
import pandas as pd
import arcpy
from arcpy.sa import *
import pathlib
import os, gdal, ogr, osr, sys
import netCDF4 #install this in arcgis environment
import json
import pickle

scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts_')
sys.path.append(scripts_dir)
sys.dont_write_bytecode = True
from functions import Preprocess

class Toolbox(object):
    def __init__(self):
        self.label = "DEM2Graph with surface"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [DEM2GRAPH]


class DEM2GRAPH(object):
    def __init__(self):
        self.label = 'DEM to Hierarchy'
        self.description = 'Extract the hierarchy of depressions from a DEM to create a graph with required attributes including area, coordinate, and curve number of basins for Rapid Flood Spreading Model'
        
    def getParameterInfo(self):
        '''define parameters of the toolbox'''
        '''parameters of the toolbox are the input dem, and the putput netcdf file name'''

        # at this step I dont define a step size
        param0 = arcpy.Parameter(displayName = "Input DEM",
            name='DEM',
            datatype='GPRasterLayer',
            parameterType='Required',
            direction='Input')
        param1 = arcpy.Parameter(displayName='Output Folder',
            name='Output workspace to store NetCDF and JSON',
            datatype='DEFolder',
            parameterType='Required',
            direction='Output')
        param2 = arcpy.Parameter(displayName='Grid of CurveNumber',
            name='CurveNumber grid',
            datatype='GPRasterLayer',
            parameterType='Required',
            direction='Input')
        param3 = arcpy.Parameter(displayName='Coordinate system',
            name='Coordinate',
            datatype='GPCoordinateSystem',
            parameterType='Optional',
            direction='Input')
        params = [param0,param1,param2, param3]
        return params
        
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return 
        


    def execute(self, params, messages):
        if params[3].altered:
            self.sr = params[3].value
        else: 
            self.sr = arcpy.SpatialReference(6343)
        arcpy.AddMessage('Creating Folder')
        arcpy.env.overwriteOutput = True
        dem_ = params[0].valueAsText
        dem = Raster(arcpy.management.ProjectRaster(dem_, "dem",self.sr))
        outfolder = params[1].valueAsText 
        cn = params[2].valueAsText
        projcn = Raster(arcpy.management.ProjectRaster(cn, "outras",self.sr))
        self.out_path = pathlib.Path(outfolder)
        os.mkdir(self.out_path)
        self.out_nc = os.path.join(outfolder, str('NetCDF.nc'))
        
        self.i = 1
        df = pd.DataFrame() #table of depressions
        V2D = pd.DataFrame() #depth volume relation tables
        road_df = pd.DataFrame() #table of depressions that are located on the studied road segment
        
        arcpy.AddMessage('Creating NetCDF file')
        sc = Preprocess(dem, str(self.out_nc))
        self.nc = sc.create_nc()
        self.resolution = sc.xsize

        while True:
            arcpy.AddMessage("--running level "+str(self.i))
            flowdir, watershed, depth, surface, bottom, newdem, max_depth = sc.dem_process(self.i, dem) 
            if max_depth == 0.0:  break
            if max_depth == -1:   break
            
            if self.i == 1:
                arcpy.env.cellSize = "MINOF"
                arcpy.AddMessage('------Getting Geometries of Sub-basins')
                geometry_tab = arcpy.sa.ZonalGeometryAsTable(watershed, "Value", 'geometry_tab')
                desc = arcpy.Describe(geometry_tab).fields
                names = [x.name for x in desc]
                arr = arcpy.da.TableToNumPyArray(geometry_tab, names)
                geometry = pd.DataFrame(arr)
                cnvalues = arcpy.ia.ZonalStatisticsAsTable(watershed, "Value", projcn, "cn_mean", "DATA", "MEAN", "CURRENT_SLICE", 90, "AUTO_DETECT")
                desc = arcpy.Describe(cnvalues).fields
                names = [x.name for x in desc]
                arr = arcpy.da.TableToNumPyArray(cnvalues, names)
                cn = pd.DataFrame(arr)
                cn = cn.rename(columns={'MEAN':'CN'})
                geometry = pd.merge(geometry, cn[['Value','CN']], how='left', right_on='Value', left_on='VALUE')
                geometry['Node'] = geometry['VALUE'].apply(lambda x: 'subbasin'+'-'+str(x))

            if self.i>1:
                arcpy.AddMessage('------get down stream')
                flow_acc = arcpy.sa.FlowAccumulation(flowdir, None, "FLOAT", "D8")
                pre_dstream = sc.get_down_inlet(pre_watershed, flow_acc)
                dep_table = sc.get_depression_table(pre_watershed,pre_depth, pre_dstream, self.i-1, watershed,pre_surface)
                df = df.append(dep_table)
            wsh = sc.get_array(watershed, fillvalue=0).astype(int)
            arcpy.AddMessage('------writing to nc')
            sc.bot_var[:,:,[self.i]] = sc.get_array(bottom)
            sc.wsh_var[:,:,[self.i]] = wsh
            sc.srf_var[:,:,[self.i]] = sc.get_array(surface)
            sc.dem_var[:,:,[self.i]] = sc.get_array(Raster(dem))
            sc.depth_var[:,:,[self.i]] = sc.get_array(depth)
            
            arcpy.AddMessage("------getting volume table")
            V2D_i = sc.get_volume_table(sc.get_array(bottom), sc.get_array(surface), wsh, sc.get_array(dem))
            V2D_i["Node"] = V2D_i["VALUE"].apply(lambda x: "L"+str(self.i)+"-"+str(x) )
            V2D= pd.concat([V2D,V2D_i], sort=False)
            self.max_l = self.i
            self.i = self.i + 1
            pre_depth = depth
            pre_watershed = watershed
            pre_surface = surface
            dem = newdem

        self.nc.close()
        df['volume-m3'] = -1*df['SUM']*(self.resolution**2) 
        df['child'] = df.child.fillna('outfall')
        last_name = np.setdiff1d(df.child.values, df.Node.values)
        print(last_name)
        source = pd.DataFrame({'Node':last_name, 'child':len(last_name)*['outfall'], 'volume-m3':len(last_name)*[-9e+15], 'AREA':len(last_name)*[0] ,'SUM':len(last_name)*[9e+15]})
        df = df.append(source, ignore_index=True)
        df.drop(np.where(df.Node.str.contains('outfall'))[0],inplace=True)
        df.loc[np.where(df.child.str.contains('outfall'))[0],'child']='outfall'
        df.loc[np.where(df.child.str.contains('\['))[0],'child']='outfall'
        df.loc[np.where(df.Node.str.contains('\['))[0],'Node']='outfall'
        df.loc[np.where(df.Node.str.contains('--'))[0],'Node']='outfall'
        df.loc[np.where(df.child.str.contains('--'))[0],'child']='outfall'
        vol = V2D[V2D.Node.isin(df.Node.values)]

        Graph_df = df.reset_index(drop=True)
        VOL2DEPTH = vol.reset_index(drop=True)
        Subbasins = geometry.reset_index(drop=True)

        writer = pd.ExcelWriter(os.path.join(outfolder, str('tables.xlsx')))
        df_list =[Graph_df, VOL2DEPTH, Subbasins,cn]
        names = ['Graph_df', 'Volume2Depth', 'Subbasins','CN']
        for i, df in enumerate(df_list):
            df.to_excel(writer,'sheet_{}'.format(names[i]))
            writer.save()
