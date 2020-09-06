import os
import sys
c_folder = os.path.abspath(os.path.dirname(__file__))
p_folder = os.path.abspath(os.path.dirname(c_folder))
sys.path.append(c_folder)
sys.path.append(p_folder)

import cv2
import json
import time
import torch
import argparse
import torch2trt
import numpy as np
import trt_pose.coco
import trt_pose.models
import PIL.Image as PILImage
import torchvision.transforms as transforms

from util import calc_feature
from util.timer import Timer
from util import recognition
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
    cmap, paf = model(processed_img)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    
    return cmap, paf, counts, objects, peaks


def draw_keypoints(origin_img, topology, cmap, paf, counts, objects, peaks):
    draw_objects = DrawObjects(topology)
    draw_objects(origin_img, counts, objects, peaks)

    return None

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


def organize_keypoints(img, counts, objects, normalized_peaks, HEIGHT, WIDTH, points_num_limit):
    count = int(counts[0])
    # normalized_points_array = []
    keypoints_set = []
    color = [(0, 255, 128), (0, 255, 255), (0, 0, 255)]

    for i in range(count):
        obj = objects[0][i]
        num_points = obj.shape[0]
        tmp_points_list = []
        for j in range(num_points):
            k = int(obj[j])
            if k < 0:
                normalized_x = 0.0
                normalized_y = 0.0
            elif k >= 0:
                peak = normalized_peaks[0][j][k]
                # x = round(float(peak[1]) * WIDTH)
                # y = round(float(peak[0]) * HEIGHT)
                normalized_x = float(peak[1])
                normalized_y = float(peak[0])
                # tmp_points_list.extend([normalized_x, normalized_y])
            tmp_points_list.append((normalized_x, normalized_y))

        if tmp_points_list.count((0.0, 0.0)) > points_num_limit:
            break

            # if tmp_points_list

        # if (len(tmp_points_list) != num_points * 2) and (len(tmp_points_list) < num_points * 2):
        #     for _ in range(int(num_points * 2 - len(tmp_points_list))):
        #         tmp_points_list.append(0.0)
        # elif len(tmp_points_list) > num_points:
        #     print(f'[INFO] the number of keypoints has been over the {num_points}')
        
        # normalized_points_array.append(tmp_points_list)
        # head_pos = [tmp_points_list[34], [tmp_points_list[35] - 20 if tmp_points_list[35] - 20 > 0 else 0][0]]
        # left_hip = [tmp_points_list[22], tmp_points_list[23]]
        # right_hip = [tmp_points_list[24], tmp_points_list[25]]
        # left_knee = [tmp_points_list[26], tmp_points_list[27]]
        # right_knee = [tmp_points_list[28], tmp_points_list[29]]
        # left_ankle = [tmp_points_list[30], tmp_points_list[31]]
        # right_ankle = [tmp_points_list[32], tmp_points_list[33]]

        """
        l_knee_to_hip_length, l_knee_to_ankle_length = recognition.hip_knee_ankle_length(left_hip, left_knee, left_ankle)
        l_btw_angle = recognition.get_angle_knee_hip_ankle(left_knee, left_hip, left_ankle)

        r_knee_to_hip_length, r_knee_to_ankle_length = recognition.hip_knee_ankle_length(right_hip, right_knee, right_ankle)
        r_btw_angle = recognition.get_angle_knee_hip_ankle(right_knee, right_hip, right_ankle)
        """
        

        # normalized_points_array.append([head_pos, l_knee_to_hip_length, l_knee_to_ankle_length, l_btw_angle, r_knee_to_hip_length, r_knee_to_ankle_length, r_btw_angle])
        # clustering_info.append([head_pos, l_knee_to_hip_length, l_knee_to_ankle_length, l_btw_angle, r_knee_to_hip_length, r_knee_to_ankle_length, r_btw_angle])
        keypoints_set.append(np.array(tmp_points_list))


        """Draw"""
        # for i, each_elem in enumerate(hip_knee_ankle):
        #     x = round(each_elem[0])
        #     y = round(each_elem[1])

        #     if (int(x), int(y)) == (0, 0):
        #         pass
            
        #     cv2.circle(img, (x, y), 2, color[int(i % 3)], 1)

    print('hello world')
    return keypoints_set


def main(args):
    """Default Variables"""
    HEIGHT = 224
    WIDTH = 224
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    global device
    device = torch.device('cuda')
    
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

    """Do conversion"""
    if not os.path.exists(args['trt_model_path']):
        print('[INFO] We are in converting step...')
        convert_from_pytorch_to_trt(model, args['trt_model_path'], HEIGHT, WIDTH)    
    
    """Set trt model"""
    model_trt = torch2trt.TRTModule()
    model_trt.load_state_dict(torch.load(args['trt_model_path']))

    # camera = RTSPCamera(width=WIDTH, height=HEIGHT, capture_width=640, capture_height=480, capture_device='rtsp://admin:intflow3121@192.168.0.103:554/cam/realmonitor?channel=1&subtype=1')

    # _check_fps = {'fps': Timer()}
    # i = 0
    img_path = '/works/GBKim_workspace/trt_pose/test_data/image_tmp/vertical.mp4_20200903_175250.698.jpg'

    img = cv2.imread(img_path)
    origin_img = cv2.resize(img, dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
        # i += 1
    start_time = time.time()
    # _check_fps['fps'].tic()

    # origin_img = camera.read()
    # copy_img = origin_img.copy()
    # print(origin_img.shape)
    # print(camera.value.shape)

    """Do inference"""
    # if (i > 1) and (i % 2 == 0):
    processed_img = preprocess(origin_img, mean, std)
    cmap, paf, counts, objects, peaks = inference(origin_img, processed_img, model_trt, topology)

    """Draw keypoints"""
    if args['draw_points']:
        draw_keypoints(origin_img, topology, cmap, paf, counts, objects, peaks)

    """Do HAR"""
    if args['do_HAR']:
        organized_keypoints_list = organize_keypoints(origin_img, counts, objects, peaks, HEIGHT, WIDTH, args['undetected_points_num_limit'])

        rmse_dict, each_diff_feature = calc_feature.get_rmse_btw_detected_objs(clustering_info=organized_keypoints_list)

    
    cv2.imshow('RTSP target frame', origin_img)

    # print(f'[INFO] counts and shape: {counts}, {counts.shape}')
    # print(f'[INFO] objectsand shape: {objects}, {objects.shape}')
    # print(f'[INFO] peaks and shape: {peaks}, {peaks.shape}')

    # key = cv2.waitKey(0) & 0xFF
    cv2.waitKey(0)
    # if key == ord('q'):
    #     break

    # _check_fps['fps'].toc()
    print('[INFO] FPS: {}'.format(1.0 / (time.time() - start_time)))
        
        
    # camera.cap.release()
    # cv2.destroyAllwindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp_address', default='rtsp://admin:intflow3121@192.168.0.103:554/cam/realmonitor?channel=1&subtype=1', type=str)
    parser.add_argument('--human_pose_json', default=os.path.join(c_folder, 'human_pose.json'), type=str)
    parser.add_argument('--pytorch_model_path', default=os.path.join(p_folder, 'pretrained_model', 'densenet121_baseline_att_256x256_B_epoch_160.pth'), help="resnet18_baseline_att_224x224_A_epoch_249.pth or densenet121_baseline_att_256x256_B_epoch_160.pth")
    parser.add_argument('--trt_model_path', default=os.path.join(p_folder, 'converted_model', 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'))
    parser.add_argument('--undetected_points_num_limit', default=5, help="Limit number of undetected points.")
    parser.add_argument('--draw_points', default=False)
    parser.add_argument('--do_HAR', default=True)
    args = parser.parse_args()
    args = vars(args)

    main(args)