CUDA_VISIBLE_DEVICES=0 python demo.py \
--config-file ../configs/ade20k/semantic-segmentation/swin/anh_maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
--input ../images \
--output ../results \
--opts MODEL.WEIGHTS ../checkpoint/model_final_v3.pth