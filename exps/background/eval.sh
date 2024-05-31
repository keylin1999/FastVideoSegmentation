export CUDA_VISIBLE_DEVICES=1

# python train.py --config resnet18_d3
# python train.py --config resnet18_d3_noBg
# python train.py --config resnet18_d4

# python train.py --config resnet34_d3
# python train.py --config resnet34_d3_noBg
# python train.py --config resnet34_d4

# python train.py --config resnet50_d3
# python train.py --config resnet50_d3_noBg
# python train.py --config resnet50_d4

# python train.py --config resnet101_d3
# python train.py --config resnet101_d3_noBg
# python train.py --config resnet101_d4

echo "------------- resnet18_d3 -------------"
python eval.py --model output/resnet18_d3/checkpoint_50000.pth

echo "------------- resnet18_d3_noBg -------------"
python eval.py --model output/resnet18_d3_noBg/checkpoint_50000.pth

echo "------------- resnet18_d4 -------------"
python eval.py --model output/resnet18_d4/checkpoint_50000.pth

echo "------------- resnet34_d3 -------------"
python eval.py --model output/resnet34_d3/checkpoint_50000.pth

echo "------------- resnet34_d3_noBg -------------"
python eval.py --model output/resnet34_d3_noBg/checkpoint_50000.pth

echo "------------- resnet34_d4 -------------"
python eval.py --model output/resnet34_d4/checkpoint_50000.pth

echo "------------- resnet50_d3 -------------"
python eval.py --model output/resnet50_d3/checkpoint_50000.pth

echo "------------- resnet50_d3_noBg -------------"
python eval.py --model output/resnet50_d3_noBg/checkpoint_50000.pth

echo "------------- resnet50_d4 -------------"
python eval.py --model output/resnet50_d4/checkpoint_50000.pth

echo "------------- resnet101_d3 -------------"
python eval.py --model output/resnet101_d3/checkpoint_50000.pth

echo "------------- resnet101_d3_noBg -------------"
python eval.py --model output/resnet101_d3_noBg/checkpoint_50000.pth

echo "------------- resnet101_d4 -------------"
python eval.py --model output/resnet101_d4/checkpoint_50000.pth