export CUDA_VISIBLE_DEVICES=2
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 0 --pixel_thresh 0 -vt 0.9 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 1 --pixel_thresh 0 -vt 0.9 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 2 --pixel_thresh 0 -vt 0.9 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 3 --pixel_thresh 0 -vt 0.9 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 4 --pixel_thresh 0 -vt 0.9 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 5 --pixel_thresh 0 -vt 0.9 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 6 --pixel_thresh 0 -vt 0.9 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 7 --pixel_thresh 0 -vt 0.9 --comment pseudo

python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 0 --pixel_thresh 0 -vt 0.9
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 1 --pixel_thresh 0 -vt 0.9
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 2 --pixel_thresh 0 -vt 0.9
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 3 --pixel_thresh 0 -vt 0.9
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 4 --pixel_thresh 0 -vt 0.9
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 5 --pixel_thresh 0 -vt 0.9
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 6 --pixel_thresh 0 -vt 0.9
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 7 --pixel_thresh 0 -vt 0.9