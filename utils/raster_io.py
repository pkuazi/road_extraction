#
# -*-coding:utf-8 -*-
#
# @Module:
#
# @Author: zhaojianghua
# @Date  : 2018-01-26 15:13
#

#!/usr/bin/env python
# -*-coding:utf-8 -*-
# created by 'root' on 18-1-25

import os, sys,time
import pandas as pd
import gdal, osr, ogr
import numpy as np
from utils.geotrans import GeomTrans
# from geotrans import GeomTrans
# from rasterio.windows import Window
# import rasterio
NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  'int64':5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}
def read_raster_by_pts(samples, feature_file, feature_name):
    '''
    :param samples:  random points
    :param feature_file:  geotiff image
    :return:
    '''

    # remove rocords with nan feature values
    daily_df = samples.dropna(axis=0, how='any')
    daily_df = pd.DataFrame(daily_df)
    # daily_df['STATION'] = daily_df.index

    sample_num = daily_df.shape[0]
    # for each monitor point, read related data. DEM data is used here
    raster = gdal.OpenShared(feature_file)
    if raster is None:
        print("Failed to open file: " + feature_file)
        sys.exit()
    feat_proj = raster.GetProjectionRef()
    # EPSG = feat_proj.GetAttrValue('AUTHORITY', 1)
    gt = raster.GetGeoTransform()

    d = gdal.Open(feature_file)
    proj = osr.SpatialReference(wkt=d.GetProjection())
    EPSG = proj.GetAttrValue('AUTHORITY', 1)

    feature_data = []
    for i in range(sample_num):
        sample_id = daily_df.iloc[i].FID
        print('sample FID is:', sample_id)

        # use loc to ge series wichi satisfying condition, and then iloc to get first element
        latitude = daily_df.iloc[i]['Lat']
        longitude = daily_df.iloc[i]['Lon']

        if EPSG != '4326':
            # transform geographic coordinates of monitor points into projected coordinate system
            inSpatialRef = osr.SpatialReference()
            inSpatialRef.SetFromUserInput("EPSG:4326")
            outSpatialRef = osr.SpatialReference()
            outSpatialRef.SetFromUserInput(feat_proj)
            transform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

            # point coordinate transformation
            geom = ogr.Geometry(ogr.wkbPoint)
            geom.AddPoint(longitude, latitude)
            geom.Transform(transform)
            longitude = geom.GetX()
            latitude = geom.GetY()

        # read feature value of the point from related data
        col = int((longitude - gt[0]) / gt[1])
        row = int((latitude - gt[3]) / gt[5])

        # print(col, row)
        if 0<col <raster.RasterXSize and 0<row < raster.RasterYSize:
            feat_value = raster.ReadAsArray(col, row, 1, 1)[0][0]
            print("the feature value is :", feat_value)
        else:
            continue

        feature_data.append([sample_id,feat_value])

    relatedata = np.array(feature_data)
    print(relatedata)
    relatedata_df = pd.DataFrame(relatedata,columns=['FID', feature_name])
    return relatedata_df

# def geom_geotransform(geom, srs_proj, dst_proj):
#     transform = GeomTrans(str(srs_proj), dst_proj)
#     geojson_crs_transformed = transform.transform_points(geom['coordinates'][0])
#     new_geom = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})
#     return new_geom

# def read_raster_by_geometry(geom, shp_proj, rasterfile):
# #     rasterize geom to mask
#     raster = rasterio.open(rasterfile, 'r')
#     dst_proj=str(raster.crs.wkt)
#     geom = geom_geotransform(geom, shp_proj, dst_proj)
   
#     bounds = geom.bounds
#     x_min = bounds[0]
#     x_max = bounds[2]
#     y_min = bounds[1]
#     y_max = bounds[3]
    
#     pixelWidth = raster.res[0]
#     pixelHeight= raster.res[1]
#     rows = int((y_max - y_min) / pixelHeight)+1
#     cols = int((x_max - x_min) /pixelWidth)+1
    
#     gt = raster.read_transform()
    
#     xoff = int((x_min-gt[0])/gt[1])
#     yoff = int((y_max -gt[3])/gt[5])

#     data = raster.read()
#     band_num = data.shape[0]
#     mask = rasterio.features.rasterize([(geom, 0)], out_shape=raster.shape, transform=raster.transform, fill=1, all_touched=True, dtype=np.uint8)
    
#     array_to_tiff(mask,dst_proj,gt, 1,'/tmp/mask.tif')
   
#     X = np.array([])
#     print(mask.max())
#     for i in range(band_num):
#         band = data[i]
#         masked_data = np.ma.array(data=band, mask=mask.astype(bool), dtype=np.float32)
#         masked_data.data[masked_data.mask] = np.nan # raster.profile nodata is 256
#         out_image = masked_data.data
#         array_to_tiff(out_image, dst_proj, (x_min, gt[1],gt[2],y_max, gt[4],gt[5]), 1, '/tmp/%s.tif'%(str(i)))
# def read_raster_by_window(path, xoff, yoff, width, height):
#     win = Window(xoff, yoff, width, height)        
#     src = rasterio.open(path, 'r')
#     gt = src.get_transform()
#     geotrans=list(gt)
#     crs = src.crs.wkt
#     imgarr = src.read((1,2,3),window = win)# read 1st, 2nd, 3rd bands from the raster in window region
#     band1=imgarr[0]
#     band2=imgarr[1]
#     band3=imgarr[2]
    
#     geotrans[0] = gt[0]+xoff*gt[1]
#     geotrans[3] = gt[3]+yoff*gt[5]

#     imgarr = np.dstack([band1,band2,band3])
# #         lab_arr = color.rgb2lab(rgb)
#     return imgarr,geotrans,crs
    
def read_raster_by_bbox(rasterfile,bbox):
    dataset = gdal.OpenShared(rasterfile)
    if dataset is None:
        print("Failed to open file: " + rasterfile)
        sys.exit(1)
    band = dataset.GetRasterBand(1)
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    proj = dataset.GetProjection()
    gt = dataset.GetGeoTransform()
    noDataValue = band.GetNoDataValue()
    datatype = band.DataType
    
    geotrans=list(gt)
    
    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    
    minx_img, maxy_img = GeomTrans( 'EPSG:4326',proj).transform_point([minx,maxy])
    maxx_img, miny_img = GeomTrans('EPSG:4326',proj).transform_point([maxx,miny])
    
    xoff = round((minx_img-gt[0])/gt[1])
    yoff = round((maxy_img-gt[3])/gt[5])
    
    geotrans[0] = gt[0]+xoff*gt[1]
    geotrans[3] = gt[3]+yoff*gt[5]
    
    xsize = int((maxx_img-minx_img)/gt[1])+1
    ysize = int((miny_img-maxy_img)/gt[5])+1

    rastervalue = band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
#     rastervalue[rastervalue == noDataValue] = -9999
    value = np.array(rastervalue,dtype='int64')
    return rastervalue, proj, geotrans
# def tif_to_jpeg(tif_filename):
#     tmp_tif = '/tmp/%s.tif'%(str(time.time()).split('.')[0])
#     cmd = 'gdal_translate -ot Byte -scale -co "TILED=YES" -co "COMPRESS=LZW" %s %s'%(tif_filename,tmp_tif)
#     os.system(cmd)
#     with rasterio.open(tmp_tif) as infile:
#         profile=infile.profile
#         raster=infile.read()
#         profile['driver']='JPEG'
#         jpeg_filename=tif_filename[:-4]+'.jpeg'
#         with rasterio.open(jpeg_filename, 'w', **profile) as dst:
#             dst.write(raster)
# #     cv2.imwrite(dst_file, data)
def array_to_tiff(data, proj, gt, dst_nbands, dst_file):
    if dst_nbands==1:
        xsize = data.shape[0]
        ysize = data.shape[1]
    else:
        xsize = data.shape[1]
        ysize = data.shape[2]
    dst_format = 'GTiff'
#     dst_nbands = 1
    # dst_datatype = gdal.GDT_Float32
    print(str(data.dtype))
    dst_datatype=NP2GDAL_CONVERSION[str(data.dtype)]

    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, dst_datatype)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
#     dst_ds.GetRasterBand(1).WriteArray(data)
    
    if dst_nbands == 1:
        dst_ds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(dst_nbands):
            dst_ds.GetRasterBand(i + 1).WriteArray(data[i,:,:])
    del dst_ds

def read_raster(rasterfile):
    dataset = gdal.OpenShared(rasterfile)
    if dataset is None:
        print("Failed to open file: " + rasterfile)
        sys.exit(1)
    band = dataset.GetRasterBand(1)
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    noDataValue = band.GetNoDataValue()

    rastervalue = band.ReadAsArray(xoff=0, yoff=0, win_xsize=xsize, win_ysize=ysize)
    rastervalue[rastervalue == noDataValue] = -9999

    return rastervalue, proj, geotrans

if __name__ == '__main__':
    image = '/mnt/win/data/xiongan/images/xa_2018098_b654.tif'
    # imgarr,gt,crs = read_raster_by_window(image, 100, 100, 1024, 1024)
    
    # array_to_tiff(imgarr, crs, gt, 3, '/tmp/test.tif')

#     # the path of data files for data fusion process
#     data_path = os.path.join(os.getcwd(),'data')
#     # the observation meteorology data from monitor station
#     samples_file = os.path.join(data_path, 'training_samples/water_samples.csv')
#     # relate data for fusion. elevation data is used here
#     feature_folder = os.path.join(data_path, 'feature_image')
# 
#     samples = pd.read_csv(samples_file, usecols=["FID","class","Lat","Lon"])
# 
#     feature_files = os.listdir(feature_folder)
#     for featurefile in feature_files:
#         if featurefile.endswith('.tif'):
#             feature_name = featurefile.split('.')[0]
#             feature_path = os.path.join(feature_folder,featurefile)
#             # read feature data according to the coordinates of the samples
#             print('the current feature name is :', feature_name)
#             feature_column = raster_read_by_pts(samples, feature_path, feature_name)
#             samples = pd.merge(samples, feature_column,on='FID')
# 
# 
#     # output the samples with feature into a csv file
#     relate_data_path = os.path.join(data_path,'sample_features.csv')
#     samples.to_csv(relate_data_path)
#     print('the data has been save into %s.'% relate_data_path)

   


