export OMP_NUM_THREADS=32
taskset -c 0-31 ./build_enhanced/FlowIntelligence \
  --output-path ./outputTest \
  --max-frames 3000 \
  --use-otsu-t1 --use-otsu-t2 --global-otsu \
  --csv-log ./outputTest/log.csv \
  --dataset-path /home/jackew/Project/FlowIntelligence/Datasets \
  --video1 T10L.mp4 --video2 T10R.mp4 \
  --stride 4 4 \  
  

  # --dataset_path /mnt/mDisk2/CityData/au_video/video \
  # --video1 63554_PAINT_R.mp4 --video2 63554_PAINT.mp4 \
  
