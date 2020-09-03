import os
import sys
c_folder = os.path.abspath(os.path.dirname(__file__))
p_folder = os.path.abspath(os.path.dirname(c_folder))
sys.path.append(c_folder)
sys.path.append(p_folder)

import cv2
import json
import torch
import argparse
import torch2trt
import trt_pose.coco
import trt_pose.models
import torchvision.transforms as transforms
import PIL.Image as PILImage

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

    return counts, objects, peaks


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
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(args['pytorch_model_path']))

    """Do conversion"""
    if not os.path.exists(args['trt_model_path']):
        print('[INFO] We are in converting step...')
        convert_from_pytorch_to_trt(model, args['trt_model_path'], HEIGHT, WIDTH)    
    
    """Set trt model"""
    model_trt = torch2trt.TRTModule()
    model_trt.load_state_dict(torch.load(args['trt_model_path']))

    camera = RTSPCamera(width=WIDTH, height=HEIGHT, capture_width=640, capture_height=480, capture_device='rtsp://admin:intflow3121@192.168.0.103:554/cam/realmonitor?channel=1&subtype=1')
    while True:
        origin_img = camera.read()
        # copy_img = origin_img.copy()
        print(origin_img.shape)
        print(camera.value.shape)

        """Do inference"""
        processed_img = preprocess(origin_img, mean, std)
        counts, objects, peaks = inference(origin_img, processed_img, model_trt, topology)

        cv2.imshow('RTSP target frame', origin_img)

        print(f'[INFO] counts and shape: {counts}, {counts.shape}')
        print(f'[INFO] objectsand shape: {objects}, {objects.shape}')
        print(f'[INFO] peaks and shape: {peaks}, {peaks.shape}')

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        
    camera.cap.release()
    cv2.destroyAllwindows()




    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp_address', default='rtsp://admin:intflow3121@192.168.0.103:554/cam/realmonitor?channel=1&subtype=1', type=str)
    parser.add_argument('--human_pose_json', default=os.path.join(c_folder, 'human_pose.json'), type=str)
    parser.add_argument('--pytorch_model_path', default=os.path.join(p_folder, 'pretrained_model', 'resnet18_baseline_att_224x224_A_epoch_249.pth'), help="resnet18_baseline_att_224x224_A_epoch_249.pth or densenet121_baseline_att_256x256_B_epoch_160.pth")
    parser.add_argument('--trt_model_path', default=os.path.join(p_folder, 'converted_model', 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'))
    args = parser.parse_args()
    args = vars(args)

    main(args)