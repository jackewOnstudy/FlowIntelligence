export OMP_NUM_THREADS=32
taskset -c 0-31 ./build/FlowIntelligence \
  --video1 T10L.mp4 --video2 T10R.mp4 \
  --use-otsu-t1 --use-otsu-t2 --global-otsu \
  --output-path ./outputTest \
  --max-frames 3000