python time_profile.py -b output/plain_50000/192_8_128_None_None_None/video_boxes.pkl | tee plain_gpu.txt
python time_profile.py -b output/plain_50000/192_8_128_None_None_None/video_boxes.pkl -d cpu | tee plain_cpu.txt

python time_profile.py -b output/adaptive_50000/192_8_128_0.9_7_0.95_0/video_boxes.pkl | tee adaptive_video_gpu.txt
python time_profile.py -b output/adaptive_50000/192_8_128_0.9_7_0.95_0/video_boxes.pkl -d cpu | tee adaptive_video_cpu.txt

python time_profile.py -b output/adaptive_single_50000/192_8_128_0.9_None_None/video_boxes.pkl | tee adaptive_single_gpu.txt
python time_profile.py -b output/adaptive_single_50000/192_8_128_0.9_None_None/video_boxes.pkl -d cpu | tee adaptive_single_cpu.txt