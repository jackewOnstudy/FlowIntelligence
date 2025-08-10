export OMP_NUM_THREADS=32
taskset -c 0-31 ./build/FlowIntelligence \
  --video1 OTCBVS1L.mp4 --video2 OTCBVS1R.mp4 \
  --output-path ./outputTest \
  --max-frames 3000 \
  # --use-otsu-t1 --use-otsu-t2 --global-otsu \