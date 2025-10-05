export OMP_NUM_THREADS=32
taskset -c 0-31 ./build/FlowIntelligence \
  --output-path ./outputTest \
  --max-frames 3000 \
  --csv-log ./outputTest/log.csv \
  --dataset-path ./FlowIntelligence/Datasets \
  --video1 T10L.mp4 --video2 T10R.mp4 \ 
  
