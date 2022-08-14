import numpy as np
import pandas as pd
import arcpy
from arcpy.sa import *
import pathlib
import os, gdal, ogr, osr

#install netCDF4 in arcgis environment
import netCDF4

class Preprocess():
    def __init__(self, tc, out_nc):
        self.output_nc = pathlib.Path(out_nc)
        self.sr = arcpy.SpatialReference(6343)
        self.dem = tc
        self.ysize = int(arcpy.GetRasterProperties_management(tc, "CELLSIZEY")[0])
        self.xsize = int(arcpy.GetRasterProperties_management(tc, "CELLSIZEX")[0])
        self.rowcount = int(arcpy.GetRasterProperties_management(tc, "ROWCOUNT")[0])
        self.colcount = int(arcpy.GetRasterProperties_management(tc, "COLUMNCOUNT")[0])
        self.xmin = np.float64(arcpy.GetRasterProperties_management(tc, "LEFT")[0])
        self.xmax = np.float64(arcpy.GetRasterProperties_management(tc, "RIGHT")[0])
        self.ymin = np.float64(arcpy.GetRasterProperties_management(tc, "BOTTOM")[0])
        self.ymax = np.float64(arcpy.GetRasterProperties_management(tc, "TOP")[0])
        self.max_l = 10
        self.step_size = 0.06
        
    def expand_one_cell (self,in_ras): 
        dist = self.xsize*np.sqrt(2)
        size = self.xsize
        out_ras = arcpy.sa.EucAllocation(in_ras, dist, None, size, "Value", None, None, "PLANAR", None, None);
        return (out_ras)

    def get_down_inlet(self, pre_watershed, accumulation):
        pour_point_wsh, pour_point_acc = self.get_pour_points(pre_watershed, accumulation)
        pour_point_wsh.save()

        # we expand the pour poin with the value of flow accumulation and find the cell in its neighbors that has a larger acc
        expand_acc = self.expand_one_cell(Raster(pour_point_acc))
        expand_wsh = self.expand_one_cell(Raster(pour_point_wsh))
        ds_inlet = Con(accumulation>expand_acc,expand_wsh)
        return (ds_inlet)        
        
    def create_nc(self):
        arcpy.AddMessage('THIS IS NEW')
        nc = netCDF4.Dataset(self.output_nc, 'w')

        # Global attributes
        nc.title = 'Hierarchy of topographic properties'

        # Create dimensions
        self.lat_dim = nc.createDimension('latitude', self.rowcount)
        self.lon_dim = nc.createDimension('longitude', self.colcount)
        self.level_dim = nc.createDimension('level', self.max_l+1)
        
        # Create variables
        self.lat_var = nc.createVariable('latitude', 'f8', ('latitude'))
        self.lat_var.standard_name = 'latitude'
        self.lat_var.axis = 'Y'

        self.lon_var = nc.createVariable('longitude', 'f8', ('longitude'))
        self.lon_var.standard_name = 'longitude'
        self.lon_var.axis = 'X'

        self.level_var = nc.createVariable('level', 'f8', ('level'))
        self.level_var.standard_name = 'level'

        self.crs_var = nc.createVariable('crs','f8', ())
        self.crs_var.standard_name = 'crs'
        self.crs_var.grid_mapping_name = 'latitude_longitude'
        self.crs_var.crs_wkt = self.sr.name

        self.bot_var = nc.createVariable('bottom', 'f8', ('latitude', 'longitude','level'),fill_value=-999)
        self.bot_var.units = 'same as the DEM units'
        self.bot_var.long_name = 'Bottom'
        self.bot_var.grid_mapping = 'crs'
        
        self.depth_var = nc.createVariable('depth', 'f8', ('latitude', 'longitude','level'),fill_value=-999)
        self.depth_var.units = 'same as the DEM units'
        self.depth_var.long_name = 'Depth'
        self.depth_var.grid_mapping = 'crs'

        self.dem_var = nc.createVariable('dem', 'f8', ('latitude', 'longitude','level'),fill_value=-999)
        self.dem_var.units = 'same as the DEM units'
        self.dem_var.long_name = 'Elevation'
        self.dem_var.grid_mapping = 'crs'

        self.wsh_var = nc.createVariable('watershed', 'u8', ('latitude', 'longitude','level'),fill_value=-9999)
        self.wsh_var.long_name = 'Watershed'
        self.wsh_var.grid_mapping = 'crs'

        self.srf_var = nc.createVariable('surface', 'f8', ('latitude', 'longitude','level'),fill_value=-999)
        self.srf_var.long_name = 'Surface'
        self.srf_var.grid_mapping = 'crs'
        
        # Load values to variables

        # Load values: latitude and longitude
        lat_values = np.arange(self.ymin, self.ymax, self.ysize)
        lon_values = np.arange(self.xmin, self.xmax, self.xsize)
        level_values = np.arange(0,self.max_l+1,1)
        self.lat_var[:] = lat_values
        self.lon_var[:] = lon_values
        self.level_var[:] = level_values
        self.nc = nc    
        return self.nc
        
    def dem_process(self,i,dem):
        try:
            tempgdb = "Level" + str(i) + ".tif"
            flow_dir = arcpy.sa.FlowDirection(dem, "NORMAL", None, "D8");
            sink = arcpy.sa.Sink(flow_dir)
            watershed = arcpy.sa.Watershed(flow_dir, sink, "Value")
            arcpy.Delete_management(sink)
            surface = arcpy.sa.ZonalFill(watershed, dem)
            bottom = arcpy.sa.ZonalStatistics(watershed, "Value", dem, "MINIMUM", "DATA", "CURRENT_SLICE")
            depth = Con(surface>dem, surface-dem, 0)
            self.next_dem = Con(Raster(surface)>Raster(dem), surface, dem)
            self.max_depth = depth.maximum
            self.next_dem.save()
            return (flow_dir, watershed, depth, surface, bottom, self.next_dem, depth.maximum)
        except arcpy.ExecuteError:
            arcpy.AddMessage('Hydrologically connected at level: '+str(i))#,  '**error text: ', arcpy.GetMessages(2))
            return ('','','','','','',-1)
            
    def get_depression_table(self, pre_watershed, pre_depth, pre_ds_inlet, i, watershed, pre_surface):
        table = arcpy.sa.ZonalStatisticsAsTable(pre_watershed, "Value", pre_depth , "out_table", "DATA", "ALL", "CURRENT_SLICE")
        desc = arcpy.Describe(table).fields
        names = [x.name for x in desc]
        arr = arcpy.da.TableToNumPyArray(table, names)
        dep = pd.DataFrame(arr)
        arcpy.AddMessage('-----getting surfaces')
        srf = arcpy.sa.ZonalStatisticsAsTable(pre_watershed, "Value", pre_surface , "out_table", "DATA", "MAXIMUM", "CURRENT_SLICE")
        desc = arcpy.Describe(srf).fields
        names = [x.name for x in desc]

        arr = arcpy.da.TableToNumPyArray(srf, names)
        srftab = pd.DataFrame(arr)
        srftab.rename(columns={'MAX':'surface'}, inplace=True)
        dep = dep.merge(srftab[['surface','Value']], how='outer',left_on='Value', right_on='Value' )
        dep['Node'] = dep['Value'].apply(lambda x: 'L'+str(i)+'-'+str(x))

        hierarchy = self.get_hierarchy(pre_watershed, pre_ds_inlet, i, watershed)
        dep_df = dep.merge(hierarchy, how = 'outer', left_on='Node', right_on='Node' )
        return dep_df
        
        
    ### getting hierarchy table
    def get_hierarchy(self,pre_watershed, ds_inlet, i, watershed):
        wsh_array = arcpy.RasterToNumPyArray(pre_watershed,nodata_to_value=-999)
        wsh_list = list(np.unique(wsh_array))
        ds_array = arcpy.RasterToNumPyArray(ds_inlet,nodata_to_value=-999)
        a = ds_array[np.where(ds_array!=-999)]
        b = wsh_array[np.where(ds_array!=-999)]
        stack = np.dstack([a,b])[0]
        hierarchy = pd.DataFrame(stack, columns = ['Node', 'child'])
        hierarchy = hierarchy.drop_duplicates(subset=['Node'])
        nl = list(set(wsh_list) - set(hierarchy.Node) - set([-999]))
        hierarchy['Node'] = hierarchy['Node'].apply(lambda x: 'L'+str(i)+'-'+str(x))
        hierarchy['child'] = hierarchy['child'].apply(lambda x: 'L'+str(i)+'-'+str(x))

        # depressions that spill to the next level
        wsh_n_arr = arcpy.RasterToNumPyArray(watershed,nodata_to_value=-999)
        w1 = wsh_array[np.where(np.isin(wsh_array, nl))]
        w2 = wsh_n_arr[np.where(np.isin(wsh_array, nl))]

        nl_hierarchy = pd.DataFrame(np.dstack([w1,w2])[0], columns = ['Node', 'child'])
        nl_hierarchy = nl_hierarchy.groupby('Node').agg(pd.Series.mode)

        nl_hierarchy = nl_hierarchy.reset_index()
        nl_hierarchy['child'] = nl_hierarchy['child'].replace(-999,'outfall')
        nl_hierarchy['Node']= nl_hierarchy['Node'].apply(lambda x: 'L'+str(i)+'-'+str(x))
        nl_hierarchy['child']= nl_hierarchy['child'].apply(lambda x: 'L'+str(i+1)+'-'+str(x))
        hierarchy = hierarchy.append(nl_hierarchy)
        print("one hierarchy achieved", len(hierarchy[hierarchy.Node.duplicated()]))
        return hierarchy
        
    def get_array(self, InRas, fillvalue=-999):
        return (arcpy.RasterToNumPyArray(InRas,nodata_to_value=fillvalue))
        
    def get_top(self,bottom_arr, surface_arr, step):
        top = np.minimum(bottom_arr + step, surface_arr)
        resid_depth = np.amax(surface_arr-top)
        return (top,resid_depth)
        
    def get_depth(self, top, dem_arr):
        depth_ = top - dem_arr
        depth = np.clip(depth_ , a_min=0, a_max=10000)
        return(depth)

    def get_volume_table(self, bottom, surface, watershed, dem):
        thresh=1
        self.l = 1
        table = pd.DataFrame(columns=['VALUE'])
        while thresh > 0:
            step = self.step_size*self.l
            top,thresh = self.get_top(bottom, surface, step)
            depth = self.get_depth(top, dem)
            table_ = self.get_volume(depth, watershed)
            table = pd.merge(table_, table , how='outer', on='VALUE')
            self.l=self.l+1
            watershed = self.update_watershed(top, surface, watershed)
        return table
        
    def get_volume(self, depth, watershed_arr):
        vol = np.bincount(np.ndarray.flatten(watershed_arr), np.ndarray.flatten(depth))
        amax = np.amax(watershed_arr)
        name = f'step{self.l}'
        table = pd.DataFrame({'VALUE':range(amax+1),name:vol*(self.xsize**2)})
        table = table[table.VALUE != 0].copy()
        return (table)
        
    def update_watershed(self, top_arr, surface_arr, watershed_arr):
        watershed_arr[top_arr == surface_arr] = 0
        return (watershed_arr)
        
    def get_pour_points(self,watershed, accumulation):
        out_ras = arcpy.ia.ZonalStatistics(watershed, "Value", accumulation, "MAXIMUM", "DATA", "CURRENT_SLICE"); 
        pp_wsh = Con(out_ras==accumulation, watershed)
        pp_acc = Con(out_ras==accumulation, accumulation)
        return(pp_wsh, pp_acc)
        