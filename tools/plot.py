from mmdet.apis import init_detector,inference_detector
from mmdet.utils import register_all_modules
import mmcv

config_file = 'P:/code/mmdetection-main/mmdetection-main/configs/train_rs/cas_sam_coco.py'
checkpoint_file = 'P:/code/mmdetection-main/mmdetection-main/tools/work_dirs/cas_sam_coco/epoch_ssdd_best.pth'
register_all_modules()
model0 = init_detector(config_file,checkpoint_file,device='cpu')

from mmdet.registry import VISUALIZERS

# init the visualizer(execute this block only once)
visualizer = VISUALIZERS.build(model0.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model0.dataset_meta

image_path = 'P:/data/ssdd_coco2017/val2017/000001.jpg'

img = mmcv.imread( image_path, channel_order='rgb')
result = inference_detector(model0,img)

# show the results
visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    wait_time=0,
)
visualizer.show()