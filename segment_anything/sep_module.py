from segment_anything.build_sam import sam_model_registry
import torch
import os

if __name__ == '__main__':
    model_type = "vit_h"
    sam_checkpoint = "P:/checkpoint/SAM/sam_vit_h.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    result_root = '../checkpoint'
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    torch.save(sam.image_encoder.state_dict(), "../checkpoint/image_encoder_"+model_type+".pth")
    torch.save(sam.prompt_encoder.state_dict(), "../checkpoint/prompt_encoder_"+model_type+".pth")
    torch.save(sam.mask_decoder.state_dict(), "../checkpoint/mask_decoder_"+model_type+".pth")