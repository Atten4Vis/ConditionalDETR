script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --batch_size 2 \
    --coco_path data/coco \
    --epochs 12 \
    --lr_drop 11 \
    --group_detr 11 \
    --output_dir output/$script_name 2>&1 &