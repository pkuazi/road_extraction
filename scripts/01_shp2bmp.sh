DIR=$(dirname $(pwd))
echo ${DIR}
RAW_DATA_DIR=$DIR/data/raw
echo $RAW_DATA_DIR

if [ -d $RAW_DATA_DIR ];then
  echo "文件夹存在"
else
  echo "文件夹不存在，创建文件夹"
  mkdir -p $RAW_DATA_DIR
fi

PROCESSED_DATA_DIR=$DIR/data/processed 
echo $PROCESSED_DATA_DIR

if [ -d $PROCESSED_DATA_DIR ];then
  echo "文件夹存在"
else
  echo "文件夹不存在，创建文件夹"
  mkdir -p $PROCESSED_DATA_DIR
fi

echo "translating shapefile mask into bmp..."
gdal_rasterize  -a label -of GTiff -a_nodata 100 -ts 11651 7768 -ot Byte $RAW_DATA_DIR/科目一样本标注_mask.shp $PROCESSED_DATA_DIR/4_maskraster.tif

echo "reclassifying the raster into 0-4"
gdal_calc.py -A $PROCESSED_DATA_DIR/4_maskraster.tif --calc="(A==101)*1+(A==102)*2+(A==103)*3+(A==104)*4" --NoDataValue=0 --outfile $PROCESSED_DATA_DIR/4_mask_.bmp

echo "finish..."
