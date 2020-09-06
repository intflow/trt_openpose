import os
import sys
c_folder = os.path.abspath(os.path.dirname(__file__))
p_folder = os.path.abspath(os.path.dirname(c_folder))
sys.path.append(c_folder)
sys.path.append(p_folder)

import cv2
import csv
import json
import tqdm
import torch
import argparse
import torch2trt
import numpy as np
import trt_pose.coco
import trt_pose.models
import torchvision.transforms as transforms
import PIL.Image as PILImage

from util import calc_feature
from imutils import paths as imutils_paths
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam_rtsp.jetcam.utils import bgr8_to_jpeg
from jetcam_rtsp.jetcam.rtsp_camera import RTSPCamera


def read_json(human_pose_json_path):
    with open(human_pose_json_path, 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    return topology, num_parts, num_links


def convert_from_pytorch_to_trt(base_model, trt_model_path, height, width):
    data = torch.zeros((1, 3, height, width)).cuda()
    model_trt = torch2trt.torch2trt(base_model, [data], fp16_mode=True, max_workspace_size=1 << 25)
    torch.save(model_trt.state_dict(), trt_model_path)

    return None


def preprocess(image, mean, std):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PILImage.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def inference(origin_img, processed_img, model, topology):
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    cmap, paf = model(processed_img)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    draw_objects(origin_img, counts, objects, peaks)
    # test = bgr8_to_jpeg(origin_img)

    return origin_img, counts, objects, peaks


def save_output_features(counts, objects, normalized_peaks, img_height, img_width, img_name, save_path):
    count = int(counts[0])
    normalized_points_array = []

    for i in range(count):
        obj = objects[0][i]
        num_points = obj.shape[0]
        tmp_points_list = []
        for j in range(num_points):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                # x = round(float(peak[1]) * img_width)
                # y = round(float(peak[0]) * img_height)
                normalized_x = float(peak[1])
                normalized_y = float(peak[0])
                # if type(normalized_x) != int:
                #     normalized_x, normalized_y = normalized_x.cpu().detach(), normalized_y.cpu().detach()
                tmp_points_list.extend([normalized_x, normalized_y])

        if (len(tmp_points_list) != num_points * 2) and (len(tmp_points_list) < num_points * 2):
            for _ in range(int(num_points * 2 - len(tmp_points_list))):
                tmp_points_list.append(0.0)
        elif len(tmp_points_list) > num_points:
            print(f'[INFO] the number of keypoints has been over the {num_points}')

        normalized_points_array.append(tmp_points_list)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_name = os.path.splitext(img_name)[0]
    with open(os.path.join(save_path, img_name + '.csv'), 'a') as f:
        for row in normalized_points_array:
            for x in row:
                f.write(str(x) + ',')
            f.write('\n')

    return normalized_points_array


def organize_keypoints(counts, objects, normalized_peaks, HEIGHT, WIDTH, points_num_limit):
    count = int(counts[0])
    # normalized_points_array = []
    keypoints_set = []
    # color = [(0, 255, 128), (0, 255, 255), (0, 0, 255)]

    for i in range(count):
        obj = objects[0][i]
        num_points = obj.shape[0]
        tmp_points_list = []
        for j in range(num_points):
            k = int(obj[j])
            if k < 0:
                x = 0.0
                y = 0.0
            elif k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * WIDTH)
                y = round(float(peak[0]) * HEIGHT)
                # normalized_x = float(peak[1])
                # normalized_y = float(peak[0])
                # tmp_points_list.extend([normalized_x, normalized_y])
            tmp_points_list.append((x, y))

        if tmp_points_list.count((0.0, 0.0)) > points_num_limit:
            break

        keypoints_set.append(np.array(tmp_points_list))

    return keypoints_set

    

def main(args):
    """Default Variables"""
    HEIGHT = 224
    WIDTH = 224
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    if os.path.split(args['pytorch_model_path'])[-1].split('_')[0] != os.path.split(args['trt_model_path'])[-1].split('_')[0]:
        print("You have to make args['pytorch_model_path'] and args['trt_model_path'] same")

    global device
    device = torch.device('cuda')

    """Create Save Path"""
    if not os.path.exists(args['result_save_path']):
        os.makedirs(os.path.join(args['result_save_path']))
    if not os.path.exists(os.path.join(args['result_save_path'], 'img')):
        os.makedirs(os.path.join(args['result_save_path'], 'img'))
    if not os.path.exists(os.path.join(args['result_save_path'], 'label')):
        os.makedirs(os.path.join(args['result_save_path'], 'label'))
    if not os.path.exists(os.path.join(args['result_save_path'], 'label2')):
        os.makedirs(os.path.join(args['result_save_path'], 'label2'))
    if not os.path.exists(os.path.join(args['result_save_path'], 'label3')):
        os.makedirs(os.path.join(args['result_save_path'], 'label3'))

    """Read human pose json file"""
    topology, num_parts, num_links = read_json(args['human_pose_json'])

    """Load base model"""
    model_name = os.path.split(args['pytorch_model_path'])[-1]
    if model_name == "resnet18_baseline_att_224x224_A_epoch_249.pth":
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    elif model_name == 'densenet121_baseline_att_256x256_B_epoch_160.pth':
        model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    else:
        print('You need to use model name correctly...')
        return None

    model.load_state_dict(torch.load(args['pytorch_model_path']))

    """Do conversion"""
    if not os.path.exists(args['trt_model_path']):
        print('[INFO] We are in converting step...')
        convert_from_pytorch_to_trt(model, args['trt_model_path'], HEIGHT, WIDTH)    
    
    """Set trt model"""
    model_trt = torch2trt.TRTModule()
    model_trt.load_state_dict(torch.load(args['trt_model_path']))


    img_list = list(imutils_paths.list_images(args['test_path_list']))
    for i, each_img in enumerate(tqdm.tqdm(img_list)):
        file_path = os.path.split(each_img)[-1]
        img_name = os.path.splitext(file_path)[0]

        img = cv2.imread(each_img)
        img = cv2.resize(img, dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
        print(f'[INFO] original shape: {img.shape}')

        """Do inference"""
        processed_img = preprocess(img, mean, std)
        img, counts, objects, normalized_peaks = inference(img, processed_img, model_trt, topology)

        """save features"""
        save_output_features(counts, objects, normalized_peaks, HEIGHT, WIDTH, file_path, os.path.join(args['result_save_path'], 'label'))

        """Get organized_keypoints and save"""
        organized_points = organize_keypoints(counts, objects, normalized_peaks, HEIGHT, WIDTH, 7)
        
        # with open(os.path.join(args['result_save_path'], 'label2', img_name + '_organized_points.csv'), 'a') as f:
        #     for row in organized_points:
        #         for x in row:
        #             f.write(str(x) + ',')
        #         f.write('\n')

        """Get RMSE"""
        rmse_dict, each_dff_feature = calc_feature.get_rmse_btw_detected_objs(clustering_info=organized_points)
        with open(os.path.join(args['result_save_path'], 'label2', img_name + '_rmse_dict.csv'), 'w') as f:
            writer = csv.writer(f)
            for key, value in rmse_dict.items():
                writer.writerow([key, value])
            # for row in rmse_dict:
            #     for x in row:
            #         f.write(str(x) + ',')
            #     f.write('\n')

        """save inference result"""
        cv2.imwrite(os.path.join(args['result_save_path'], 'img', file_path), img)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--rtsp_address', default='rtsp://admin:intflow3121@192.168.0.103:554/cam/realmonitor?channel=1&subtype=1', type=str)
    parser.add_argument('--result_save_path', default=os.path.join(c_folder, 'results', '20200906'))
    parser.add_argument('--human_pose_json', default=os.path.join(c_folder, 'human_pose.json'), type=str)
    parser.add_argument('--test_path_list', default=os.path.join(p_folder, 'test_data', 'image'))
    parser.add_argument('--pytorch_model_path', default=os.path.join(p_folder, 'pretrained_model', 'densenet121_baseline_att_256x256_B_epoch_160.pth'), help="resnet18_baseline_att_224x224_A_epoch_249.pth or densenet121_baseline_att_256x256_B_epoch_160.pth")
    parser.add_argument('--trt_model_path', default=os.path.join(p_folder, 'converted_model', 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'))
    args = parser.parse_args()
    args = vars(args)

    main(args)