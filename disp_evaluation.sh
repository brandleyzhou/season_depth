MODEL=ft_diffnet_mr_19
#PATH="$HOME/anaconda3/envs/manydepth/bin/:$PATH"
#LD_LIBRARY_PATH="$HOME/anaconda3/envs/manydepth/lib:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH
#python test_simple.py --image_path data_path/SeasonDepth_testset/images --model_name $MODEL
python test_simple.py --image_path data_path/submission --model_name $MODEL

#PATH="$HOME/anaconda3/bin/:$PATH"
#python evaluation.py  --pred_pth $MODEL --gt_pth data_path/seasondepth/slice0/depth
#rm -fr $MODEL

#python evaluation.py  --pred_pth val_depth_predictions --gt_pth data_path/SeasonDepth_testset/depth
#python evaluation.py --pred_pth val_depth_predictions --gt_pth data_path/SeasonDepth_testset/depth
