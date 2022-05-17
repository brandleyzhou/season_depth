# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

env_list = ['env00', 'env01', 'env02', 'env03', 'env04', 'env05', 'env06', 'env07', 'env08', 'env09', 'env10', 'env11']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    #encoder = networks.ResnetEncoder(18, False)
     
    encoder = networks.test_hr_encoder.hrnet18(True)
    encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.HRDepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    paths = []
    for env in env_list:
        for cam in ['c0','c1']:
            images = glob.glob(os.path.join(args.image_path, env, cam, '*.{}'.format(args.ext)))
            for i in images:
                paths.append(i)
    output_directory = args.model_name
    with open('valset.txt','w') as f:
        for i in paths:
            f.write(i + '\n')


    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            output_name = image_path[-35:-4]
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            topil = transforms.ToPILImage()
            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            min_depth = 0.1 
            max_depth = 68500.0
            #pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)
            disp = 1 / outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)


            # Saving greyscale depth image
            disp_resized_np =  disp_resized.squeeze().cpu().numpy()
            
            disp_resized_np = (disp_resized_np - disp_resized_np.min()) / (disp_resized_np.max() - disp_resized_np.min()) 
            im = pil.fromarray((disp_resized_np * 65536).astype(np.uint16))
            #im = pil.fromarray((disp_resized_np).astype(np.uint16))
            if not os.path.exists(os.path.join(output_directory, 'slice2')):
                os.makedirs(os.path.join(output_directory, 'slice2'))
            name_dest_im = os.path.join(output_directory, 'slice2',"{}.png".format(output_name))
            im.save(name_dest_im)

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)