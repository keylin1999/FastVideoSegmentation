sh scripts/exps/one_patch_size.sh | tee scripts/output/one_patch_size/eval.txt
sh scripts/exps/one_patch_cal_reduction | tee scripts/output/one_patch_size/reduction.txt

sh exps/tile_size/tiled_tile_size.sh | tee exps/tile_size/eval.txt
sh exps/tile_size/cal_reduction.sh | tee exps/tile_size/reduction.txt

sh exps/adaptive_param/eval.sh | tee exps/adaptive_param/eval.txt
sh exps/adaptive_param/reduction.sh | tee exps/adaptive_param/reduction.txt

sh exps/video_segmentation/eval.sh | tee exps/video_segmentation/eval.txt
sh exps/video_segmentation/eval1.sh | tee exps/video_segmentation/eval1.txt
sh exps/video_segmentation/eval2.sh | tee exps/video_segmentation/eval2.txt


sh exps/video_segmentation/reduction_pseduo.sh | tee sh exps/video_segmentation/reduction_pseduo.txt
sh exps/video_segmentation/reduction_real.sh | tee sh exps/video_segmentation/reduction_real.txt