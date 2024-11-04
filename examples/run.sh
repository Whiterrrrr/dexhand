export export CUDA_VISIBLE_DEVICES=0
python segmentation/train_segmentation.py --use_img --cat laptop --run half --arch pn --half
# python segmentation/train_segmentation.py --use_img --cat bucket --run full --arch pn
# python segmentation/train_segmentation.py --use_img --cat bucket --run full --arch mpn
# python segmentation/train_segmentation.py --use_img --cat bucket --run full --arch lpn
# python reconstruction/train_reconstruction.py --use_img --cat faucet --run full
# python simsiam/train_simsiam.py --use_img --cat toilet --run full
# python train_segmentation.py --use_img --cat laptop --run 0 --arch pn --vis pn_0.pth  # visualize the result