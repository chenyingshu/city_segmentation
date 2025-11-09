RES=4096
for SCENE in  SG_Downtown_large_ctr3_packed
do
    CUDA_VISIBLE_DEVICES=1 python demo_split_merge.py \
    --config-file ../configs/ade20k/semantic-segmentation/swin/anh_maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
    --input ../../3d_aware_style_transfer/outputs/planned_views_v4/${SCENE}/images_${RES} \
    --output ../../3d_aware_style_transfer/dataset/segmentation/${SCENE} \
    --opts MODEL.WEIGHTS ../checkpoint/model_final_v3.pth
done
