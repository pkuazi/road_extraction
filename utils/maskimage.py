# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:01:00 2020

@author: ink
"""

from __future__ import print_function
import os,json
import fiona
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from affine import Affine
from shapely.geometry import shape
from shapely.geometry import mapping, Polygon
from geotrans import GeomTrans
from fiona.crs import from_epsg
from scipy.stats import kurtosis, skew
# from shp2json import Shp2Json
# from utils.raster_io import  array_to_tiff
import gdal,ogr
import numpy.ma as ma

np.seterr(divide='ignore', invalid='ignore')

class Shp2Json:
    def __init__(self, shapefile):
        self.shapefile = shapefile

    def shp2json_fiona(self):
        vector = fiona.open(self.shapefile, 'r')
        geomjson_list = []
        for feature in vector:
            # create a shapely geometry
            # this is done for the convenience for the .bounds property only
            # feature['geoemtry'] is in Json format
            geojson = feature['geometry']
            geomjson_list.append(geojson)
        return geomjson_list

    def shp2json_ogr(self):
        dr = ogr.GetDriverByName("ESRI Shapefile")
        shp_ds = dr.Open(self.shapefile)
        layer = shp_ds.GetLayer(0)
        # shp_proj = layer.GetSpatialRef()
        # shp_proj4 = shp_proj.ExportToProj4()
        # extent = layer.GetExtent()  # minx, maxx, miny,  maxy
        geomjson_list = []
        feat_num = layer.GetFeatureCount()
        for i in range(feat_num):
            feat = layer.GetFeature(i)
            geom = feat.GetGeometryRef()
            geojson = json.loads(geom.ExportToJson())
            geomjson_list.append(geojson)
        return geomjson_list
    
def array_to_tiff(data, proj, gt, dst_nbands, dst_file):
    xsize = data.shape[0]
    ysize = data.shape[1]
    dst_format = 'GTiff'
#     dst_nbands = 1
    dst_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, dst_datatype)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
#     dst_ds.GetRasterBand(1).WriteArray(data)
    
    if dst_nbands == 1:
        dst_ds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(dst_nbands):
            dst_ds.GetRasterBand(i + 1).WriteArray(data[:,:,i])
    del dst_ds
class MaskImage:
    def __init__(self, geomjson=None, geomproj=None, rsimage=None):
        self.geomjson = geomjson
        self.geomproj = geomproj
        self.raster = rsimage
        transform = GeomTrans(str(self.geomproj), str(self.raster.crs.wkt))
        if geomjson['type']=='Polygon':
            geojson_crs_transformed = transform.transform_points(self.geomjson['coordinates'][0])
        elif geomjson['type']=='LineString':
            geojson_crs_transformed = transform.transform_points(self.geomjson['coordinates'])
        print(geojson_crs_transformed)
        geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})
        self.geometry = geometry

    def get_data(self):
        shifted_affine = self._get_transform()
        window, mask = self._get_mask(shifted_affine)

        bands_num = self.raster.count
        multi_bands_data = []
        for i in range(bands_num):
            # band_name = raster.indexes[i]
            data = self.raster.read(i + 1, window=window)

            # create a masked numpy array
            masked_data = np.ma.array(data=data, mask=mask.astype(bool), dtype=np.float32)
            masked_data.data[masked_data.mask] = np.nan  # raster.profile nodata is 256
            out_image = masked_data.data
            multi_bands_data.append(out_image)
        out_data = np.array(multi_bands_data)
        return out_data

    def _get_transform(self):
        bbox = self.raster.bounds
        extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
        raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

        if not self.geometry.intersects(raster_boundary):
            print('shape do not intersect with rs image')
            return

        # get pixel coordinates of the geometry's bounding box,
        ll = self.raster.index(*self.geometry.bounds[0:2])  # lowerleft bounds[0:2] xmin, ymin
        ur = self.raster.index(*self.geometry.bounds[2:4])  # upperright bounds[2:4] xmax, ymax

        # create an affine transform for the subset data
        t = self.raster.transform
        shifted_affine = Affine(t.a, t.b, t.c + ll[1] * t.a, t.d, t.e, t.f + ur[0] * t.e)
        return shifted_affine

    def _get_mask(self, shifted_affine):
        bbox = self.raster.bounds
        extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
        raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

        if not self.geometry.intersects(raster_boundary):
            print('shape do not intersect with rs image')
            return

        # get pixel coordinates of the geometry's bounding box,
        ll = self.raster.index(*self.geometry.bounds[0:2])  # lowerleft bounds[0:2] xmin, ymin
        ur = self.raster.index(*self.geometry.bounds[2:4])  # upperright bounds[2:4] xmax, ymax

        # read the subset of the data into a numpy array
        window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))
        data = self.raster.read(1, window=window)
        mask = rasterio.features.rasterize([(self.geometry, 0)], out_shape=data.shape, transform=shifted_affine, fill=1,
                                           all_touched=True, dtype=np.uint8)
        return window, mask

def geom_geotransform(geom, srs_proj, dst_proj):
    transform = GeomTrans(str(srs_proj), dst_proj)
    geojson_crs_transformed = transform.transform_points(geom['coordinates'][0])
    new_geom = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})
    return new_geom

def mask_band_by_geom(geom, bandarray, gt):
#     rasterize geom to mask
#     dst_proj=str(crs_wkt)
#     geom = geom_geotransform(geom, shp_proj, dst_proj)
    
    affine_gt = Affine(gt[1],gt[2], gt[0],gt[4], gt[5],gt[3])#Affine(30.0, 0.0, 382635.0,0.0, -30.0, 4335885.0)  -- [382635.0, 30.0, 0.0, 4335885.0, 0.0, -30.0]
    mask = rasterio.features.rasterize([(geom, 0)], out_shape=bandarray.shape, transform=affine_gt, fill=1, all_touched=True, dtype=np.uint8)
    
    masked_data = np.ma.array(data=bandarray, mask=mask.astype(bool), dtype=np.float32)
    masked_data.data[masked_data.mask] = np.nan # raster.profile nodata is 256
    out_image = masked_data.data
  
    return out_image
          

def mask_image_by_geojson_polygon(geojson_polygon, geoproj, rasterfile):
    '''

    :param geojson_polygon: the geojson format of a polygon
    :param geoproj: the projection coordinate system of the input polygon
    :param raster:  the raster data after executing the raster = rasterio.open(raster_image_file, 'r')
    :return: the data cut out from the raster by the polygon, and its geotransformation
    '''
    raster = rasterio.open(rasterfile, 'r')
    transform = GeomTrans(str(geoproj), str(raster.crs.wkt))
    geojson_crs_transformed = transform.transform_points(geojson_polygon['coordinates'][0])
    geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})

    bbox = raster.bounds
    extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
    raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

    # if not geometry.intersects(raster_boundary):
    #     return
    if not geometry.within(raster_boundary):
        print('the geometry is not within the raster image')
        return

    # get pixel coordinates of the geometry's bounding box,
    ll = raster.index(*geometry.bounds[0:2])  # lowerleft bounds[0:2] xmin, ymin
    ur = raster.index(*geometry.bounds[2:4])  # upperright bounds[2:4] xmax, ymax

    # create an affine transform for the subset data
    t = raster.transform
    shifted_affine = Affine(t.a, t.b, t.c + ll[1] * t.a, t.d, t.e, t.f + ur[0] * t.e)

    # read the subset of the data into a numpy array
    window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))
    bands_num = raster.count
    multi_bands_data = []
    for i in range(bands_num):
        # band_name = raster.indexes[i]
        data = raster.read(i + 1, window=window)
        # rasterize the geometry
        mask = rasterio.features.rasterize([(geometry, 0)], out_shape=data.shape, transform=shifted_affine, fill=1,
                                           all_touched=True, dtype=np.uint8)

        # create a masked numpy array
        masked_data = np.ma.array(data=data, mask=mask.astype(bool), dtype=np.float32)
        masked_data.data[masked_data.mask] = np.nan  # raster.profile nodata is 256
        out_image = masked_data.data
        multi_bands_data.append(out_image)
    out_data = np.array(multi_bands_data)
    return out_data, shifted_affine

#定义一个求三阶颜色矩的函数
def var(x=None):
    mid = np.mean(((x - x.mean()) ** 3))
    return np.sign(mid) * abs(mid) ** (1/3)
def object_feats(data):
    # min, max, mean, variance, skewness, kurtosis
    mini = np.nanmin(ma)
    maxi= np.nanmax(ma)
    avg = np.nanmean(data[np.isfinite(data)])
    var = np.nanvar(data[np.isfinite(data)])
    std=np.nanstd(data[np.isfinite(data)])
    sc = np.nanmean((data[np.isfinite(data)] - avg) ** 3)  #计算偏斜度
    sk = np.nanmean((data[np.isfinite(data)] - avg) ** 4) / pow(var, 2) #计算峰度

    #     rd = masked_arr[0]
#     gd = masked_arr[1]
#     bd = masked_arr[2]
#     x=rd[rd!=0].mean()
#     feat.append(rd[rd!=0].mean())
#     feat.append(gd[gd!=0].mean())
#     feat.append(bd[bd!=0].mean())
#     feat.append(rd[rd!=0].std())
#     feat.append(gd[gd!=0].std())
#     feat.append(bd[bd!=0].std())
#     feat.append(var(rd[rd!=0]))
#     feat.append(var(gd[gd!=0]))
#     feat.append(var(bd[bd!=0]))
    print( avg, var, std)
    # return [ avg, var, std,sc,sk]
    return [ avg, std,sc,sk]
def mask_feats_by_geometry(poly_coords, raster):
    import time
    T1 = time.perf_counter()
    
    transform = GeomTrans(str('EPSG:4326'), str(raster.crs.wkt))
    geojson_crs_transformed = transform.transform_points(poly_coords)
    geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})

    bbox = raster.bounds
    extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
    raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

    if not geometry.intersects(raster_boundary):
        return (None, None)

    # get pixel coordinates of the geometry's bounding box,
    ll = raster.index(*geometry.bounds[0:2])  # lowerleft bounds[0:2] xmin, ymin
    ur = raster.index(*geometry.bounds[2:4])  # upperright bounds[2:4] xmax, ymax

    # # create an affine transform for the subset data
    # t = raster.transform
    # shifted_affine = Affine(t.a, t.b, t.c + ll[1] * t.a, t.d, t.e, t.f + ur[0] * t.e)
    #
    # # read the subset of the data into a numpy array
    # window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))

    # when the shapefile polygon is larger than the raster
    row_begin = ur[0] if ur[0] > 0 else 0
    row_end = ll[0] + 1 if ll[0] > -1 else 0
    col_begin = ll[1] if ll[1] > 0 else 0
    col_end = ur[1] + 1 if ur[1] > -1 else 0
    window = ((row_begin, row_end),(col_begin, col_end))
    
    out_data = raster.read(window=window)
    
    # create an affine transform for the subset data
    t = raster.transform
    shifted_affine = Affine(t.a, t.b, t.c + col_begin * t.a, t.d, t.e, t.f + row_begin * t.e)
    
#     out_shape = ((window[0][1]-window[0][0]), (window[1][1]-window[1][0]))
    out_shape=out_data.shape[1:3]
    mask = rasterio.features.rasterize([(geometry, 0)], out_shape=out_shape, transform=shifted_affine, fill=1,all_touched=True, dtype=np.uint8)
    print(mask.max())
    cv2.imwrite('/tmp/mask.jpg',mask)
    bands_list=[]
    
    # band_data = out_data
    # ply_data = band_data*mask
    
    for i in range(out_data.shape[0]):
        print(i)
        band_data = out_data[i]
        # print(band_data)
        # cv2.imwrite('/tmp/bai_data.jpg',band_data)
        ply_data = ma.masked_array(band_data, mask=mask)
        ply_data.data[ply_data.mask] = np.nan
        # ply_data = band_data*mask
        # print(ply_data)
        cv2.imwrite('/tmp/mask_data.jpg',ply_data.data)
        feats=object_feats(ply_data.data)
        bands_list.extend(feats)
        
        # feat.append(ply_data.mean())
        # feat.append(ply_data[ply_data!=0].mean())
    T2 =time.perf_counter()
    deltat=((T2 - T1)*1000)
    # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    return bands_list, deltat
#     masked_arr = np.array(bands_list)
# #     print(masked_arr)
#     feat=[]

    # return feat
def main():
    polygons={}
    # rgb_dir = "/mnt/win/data/Potsdam/2_Ortho_RGB"
    # osm_dir = '/mnt/win/data/Potsdam/8_OSM_buildings/'
    feat_dir = '/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/12_ndwi/'
    # feat_dir='/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/10_xception_feature/'
    # feat_dir = '/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/11_NDWI_B'
    # osm_dir = '/tmp/test1.shp'
    osm_dir = '/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/8_OSM_buildings'
    rsts = list(filter(lambda x: x.startswith('top_potsdam_2_10') and x.endswith('indices.tif'), os.listdir(feat_dir))) 
    print(rsts)
    wholetime=0
    i=0
    for rastername in rsts:
        # print(rastername)
#     rasterfile = "/mnt/win/data/Potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif"
        rasterfile = os.path.join(feat_dir,rastername)
        import time
        T1 = time.perf_counter()
        raster = rasterio.open(rasterfile, 'r')
        T2 =time.perf_counter()
        deltat=((T2 - T1)*1000)
        print('io程序运行时间:%s毫秒' %deltat)
        wholetime+=deltat
        r=rastername.split('_')[2]
        c=rastername.split('_')[3]
        # shapefile=os.path.join(osm_dir,'osm_buildings_%s_%s.shp'%(r,c))
        shapefile = os.path.join(osm_dir, 'osm_buildings.shp')
        # shapefile = '/tmp/test1.shp'
#     shapefile = "/mnt/win/data/Potsdam/8_OSM_buildings/osm_buildings_2_10.shp"
        shp2json = Shp2Json(shapefile)
        geojson_list = shp2json.shp2json_fiona()
        
        vector = fiona.open(shapefile, 'r')
        print(rasterfile, shapefile)
        for feature in vector:
            osm_id=feature['properties']['osm_id']
            # if osm_id != '136893614':
            #     continue
            true_label=feature['properties']['label']
            geojson = feature['geometry']
            
            # if multipolygon, split it into polygons
            if geojson['type']=='MultiPolygon':
                num_plys=len(geojson['coordinates'])
                for i in range(num_plys):
                    new_osm_id = str(osm_id)+'_'+str(i)
                    geojson = feature['geometry']
                    poly_coords=geojson['coordinates'][i][0]
                    
                    # feats = mask_feats_by_geometry(geometry, raster)
                    feats, deltat= mask_feats_by_geometry(poly_coords, raster)
                    if feats is None:
                        continue
                    
                    wholetime+=deltat
                    i+=1
                    polygons[new_osm_id]={}
                    polygons[new_osm_id]['geometry']={'coordinates': [poly_coords], 'type': 'Polygon'}
                    polygons[new_osm_id]['features']=feats 
                    polygons[new_osm_id]['true_label']=true_label
                    
            elif geojson['type']=='Polygon':
                poly_coords=geojson['coordinates'][0]
                
                # feats = mask_feats_by_geometry(poly_coords, raster)
                feats, deltat= mask_feats_by_geometry(poly_coords, raster)
                # if feats is None:
                #     continue
                if feats is None:
                    continue
                    
                wholetime+=deltat
                i+=1
                polygons[osm_id]={}
                polygons[osm_id]['geometry']=geojson
                polygons[osm_id]['features']=feats 
                polygons[osm_id]['true_label']=true_label
    # print(polygons)
    avgtime=wholetime/i
    print('read raster data of %s osm polygon by rasterio, average time is:'%(i),avgtime)
    
    #save osm_id and raster features into csv
    df = pd.DataFrame()
    osmids=list(polygons.keys())
    for i in range(len(osmids)):
        geomid = osmids[i]
        feats=polygons[geomid]['features']
        feats.append(polygons[geomid]['true_label'])
        df[geomid]=feats
        print(feats)
        # df = df.append(pd.DataFrame({'osm_id':geomid,'feats':feats},index = [i]),ignore_index=True) 
    df=df.transpose()
    df.to_csv('/tmp/potsdam_2_10_indices.csv', index=True)
    return feats
            
    # save polygons with raster features into shpfile
    # shpdst='/tmp/test.shp'
    # schema = {
    #     'geometry': 'Polygon',
    #     'properties': {'id': 'int','osm_id':'str','ravg':'float','gavg':'float','bavg':'float','rstd':'float','gstd':'float','bstd':'float','rvar':'float','gvar':'float','bvar':'float'},
    # }
    # with fiona.open(shpdst, 'w', 'ESRI Shapefile', schema, crs=from_epsg(4326)) as c:
    #     ## If there are multiple geometries, put the "for" loop here
    #     n=0
    #     for geomid in polygons.keys():
    #         geojson =polygons[geomid]['geometry']
    #         coordinates = geojson['coordinates']
    #         poly = Polygon(coordinates[0])
    #         feats=polygons[geomid]['features']
    #         print(geomid,feats)
    #         c.write({
    #             'geometry': mapping(poly),
    #             'properties': {'id':n,'osm_id':geomid,'ravg':feats[0],'gavg':feats[1],'bavg':feats[2],'rstd':feats[3],'gstd':feats[4],'bstd':feats[5],'rvar':feats[6],'gvar':feats[7],'bvar':feats[8]}
    #         })
    #         n=n+1
def mask_dbox_by_geometry(poly_coords,dboxds):
    bands_list=[]
    feat=[]
    import time
    T1 = time.perf_counter()
    
    transform = GeomTrans(str('EPSG:4326'), dboxds.GetProjectionRef())
    geojson_crs_transformed = transform.transform_points(poly_coords)
    geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})

    bbox = dboxds.GetExtent()
    extent = [[bbox[0], bbox[3]], [bbox[0], bbox[2]], [bbox[1], bbox[2]], [bbox[1], bbox[3]]]
    raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

    if not geometry.intersects(raster_boundary):
        return (None, None)
    
    raster_gt = dboxds.GetGeoTransform()
    # (366976.5, 0.05, 0.0, 5808562.6, 0.0, -0.05)
    # col_begin=round((geometry.bounds[0]-raster_gt[0])/raster_gt[1]) if geometry.bounds[0]>raster_gt[0] else 0
    # row_begin=round((geometry.bounds[3]-raster_gt[3])/raster_gt[5]) if geometry.bounds[3]<raster_gt[3] else 0
    winx = round((geometry.bounds[2]-geometry.bounds[0])/raster_gt[1])
    winy = round((geometry.bounds[1]-geometry.bounds[3])/raster_gt[5] )
    
    geom_gt = (geometry.bounds[0],raster_gt[1],raster_gt[2],geometry.bounds[3],raster_gt[4],raster_gt[5])
    if winx<=2000 and winy<=2000:
        # print(winx, winy)
        out_data = dboxds.ReadRegion(geom_gt, winx, winy)

        shifted_affine = Affine(raster_gt[1], raster_gt[2],geometry.bounds[0], raster_gt[4],raster_gt[5],geometry.bounds[3])
    #     out_shape = ((window[0][1]-window[0][0]), (window[1][1]-window[1][0]))
        # out_shape=out_data.shape[1:3]
        out_shape=out_data.shape
        mask = rasterio.features.rasterize([(geometry, 1)], out_shape=out_shape, transform=shifted_affine, fill=0,all_touched=True, dtype=np.uint8)
        ply_data = out_data*mask
        
        # for i in range(out_data.shape[0]):
        #     band_data = out_data[i]
        #     ply_data = band_data*mask
        #     bands_list.append(ply_data)
            
            # feat.append(ply_data.mean())
            # feat.append(ply_data[ply_data!=0].mean())
    else:
        # Define a polygon feature geometry with one attribute
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }
        
        # Write a new Shapefile
        with fiona.open('/tmp/test.geojson', 'w', 'GeoJSON', schema,crs=dboxds.GetProjectionRef()) as c:
            ## If there are multiple geometries, put the "for" loop here
            c.write({
                'geometry': mapping(geometry),
                'properties': {'id': 0},
            })
            
        tile_list = dboxds.QueryTiles('/tmp/test.geojson')
        ply_data=[]
        for x,y in tile_list:
            tile_gt,out_data = dboxds.MaskTile(x,y,'/tmp/test.geojson')
            # tile_gt,tile_size = dboxds.GetTileInfo(x,y)
            
            shifted_affine = Affine(tile_gt[1], tile_gt[2],tile_gt[0], tile_gt[4],tile_gt[5],tile_gt[3])
            out_shape=out_data.shape
            mask = rasterio.features.rasterize([(geometry, 1)], out_shape=out_shape, transform=shifted_affine, fill=0,all_touched=True, dtype=np.uint8)
            ply_data.append( out_data*mask)

    T2 =time.perf_counter()
    deltat=((T2 - T1)*1000)
    print(ply_data)
    # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    return ply_data, deltat
def dbox():
    polygons={}
    # rgb_dir = "/mnt/win/data/Potsdam/2_Ortho_RGB"
    # osm_dir = '/mnt/win/data/Potsdam/8_OSM_buildings/'
    feat_dir = '/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/11_deeplab_feature/dbox'
    # feat_dir='/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/10_xception_feature/'
    # feat_dir = '/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/11_NDWI_B'
    
    osm_dir = '/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam/8_OSM_buildings'
    rsts = list(filter(lambda x: x.startswith('top_potsdam_2_10') and x.endswith('.DBOX'), os.listdir(feat_dir))) 
    print(rsts)
    wholetime=0
    i=0
    for rastername in rsts:
        # print(rastername)
#     rasterfile = "/mnt/win/data/Potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif"
        import dboxio
        rasterfile = os.path.join(feat_dir,rastername)
        import time
        T1 = time.perf_counter()
        # raster = rasterio.open(rasterfile, 'r')
        dboxds=dboxio.Open(rasterfile)
        T2 =time.perf_counter()
        deltat=((T2 - T1)*1000)
        print('io程序运行时间:%s毫秒' %deltat)
        wholetime+=deltat
        
        # shapefile=os.path.join(osm_dir,'osm_buildings_%s_%s.shp'%(r,c))
        # shapefile = os.path.join(osm_dir, 'osm_buildings_2_10.shp')
        shapefile = '/tmp/test1.shp'
#     shapefile = "/mnt/win/data/Potsdam/8_OSM_buildings/osm_buildings_2_10.shp"
        shp2json = Shp2Json(shapefile)
        geojson_list = shp2json.shp2json_fiona()
        
        vector = fiona.open(shapefile, 'r')
        print(rasterfile, shapefile)
        for feature in vector:
            # osm_id=feature['properties']['osm_id']
            # true_label=feature['properties']['label']
            geojson = feature['geometry']
            
            # if multipolygon, split it into polygons
            if geojson['type']=='MultiPolygon':
                num_plys=len(geojson['coordinates'])
                for i in range(num_plys):
                    # new_osm_id = str(osm_id)+'_'+str(i)
                    # geojson = feature['geometry']
                    poly_coords=geojson['coordinates'][i][0]
                    
                    feats, deltat= mask_dbox_by_geometry(poly_coords,dboxds)
                    # feats = mask_feats_by_geometry(geometry, raster)
                    # feats, deltat= mask_feats_by_geometry(poly_coords, raster)
                    if feats is None:
                        continue
                    
                    wholetime+=deltat
                    i+=1
                    # polygons[new_osm_id]={}
                    # polygons[new_osm_id]['geometry']={'coordinates': [poly_coords], 'type': 'Polygon'}
                    # polygons[new_osm_id]['features']=feats 
                    # polygons[new_osm_id]['true_label']=true_label
                    
            elif geojson['type']=='Polygon':
                poly_coords=geojson['coordinates'][0]
                
                # feats = mask_feats_by_geometry(geometry, raster)
                feats, deltat= mask_dbox_by_geometry(poly_coords, dboxds)
                # if feats is None:
                #     continue
                if feats is None:
                    continue
                    
                wholetime+=deltat
                i+=1
                # polygons[osm_id]={}
                # polygons[osm_id]['geometry']=geojson
                # polygons[osm_id]['features']=feats 
                # polygons[osm_id]['true_label']=true_label
    # print(polygons)
    avgtime=wholetime/i
    print('read raster data of %s osm polygon by dobx, average time is:'%(i),avgtime)
    return feats
if __name__ == '__main__':
    import cv2
    feats1 = main()
    # cv2.imwrite('/tmp/rasterio_test1.jpg', feats1)
    # feats2 = dbox()
    # cv2.imwrite('/tmp/dbox_test1.jpg', feats2)
    # print(np.array_equal(feats1,feats2))
