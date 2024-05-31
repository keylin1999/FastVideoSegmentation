export CUDA_VISIBLE_DEVICES=2

# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 0 --pixel_thresh 23874 -vt 0
# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 0 --pixel_thresh 39138 -vt 0
# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 0 --pixel_thresh 53229 -vt 0
# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 0 --pixel_thresh 67029 -vt 0
# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box -e 50000 -at 0.90 --cont 1000 --pixel_thresh 67029 -vt 0

# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 0 --pixel_thresh 23874 -vt 0 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 0 --pixel_thresh 39138 -vt 0 --comment pseudo
python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 0 --pixel_thresh 53229 -vt 0 --comment pseudo
# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 0 --pixel_thresh 67029 -vt 0 --comment pseudo
# python video_seg.py -mo resnet18_d3 -s 192 -ma 8 -lt 128 -me adaptive -pa bounding_box_pseudo -e 50000 -at 0.90 --cont 1000 --pixel_thresh 67029 -vt 0 --comment pseudo