DIR=$(dirname $(pwd))
PROCESSED_DATA_DIR=$DIR/data/processed 

if [ -d $PROCESSED_DATA_DIR ];then
  echo "文件夹存在"
else
  echo "文件夹不存在，创建文件夹"
  mkdir -p $PROCESSED_DATA_DIR
fi

echo 'generating tiles for trainset, valset, and testset...'
python $DIR/dataloaders/split_train_val_test.py --list_dstpath=$PROCESSED_DATA_DIR 
