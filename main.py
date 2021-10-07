import os
import argparse
#import project_image_without_optimizer as projector   # much faster but effect worse
import project_image as projector
from encoder.model import BiSeNet
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
from PIL import ImageFilter
import cv2
from tools.face_alignment import image_align
from tools.landmarks_detector import LandmarksDetector
from tools import functions

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    mask = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]), dtype=np.uint8)
    num_of_class = np.max(vis_parsing_anno)

    idx = 11
    for pi in range(1, num_of_class + 1):
        index = np.where((vis_parsing_anno <= 5) & (
            vis_parsing_anno >= 1) | ((vis_parsing_anno >= 10) & (vis_parsing_anno <= 13)))
        mask[index[0], index[1]] = 1
    return mask


def main():
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    """
    parser = argparse.ArgumentParser(description='Model Face Swap', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_img', type=str, default='input/test1.jpg', help='Directory with raw images for face swap')
    parser.add_argument('--output_dir', type=str, default='output/', help='Directory for storing changed images')
    parser.add_argument('--project_style', type=str, default='model', help='model/pop-star/kids/wanghong...')
    parser.add_argument('--record', type=bool, default=True, help='Recording process')
    parser.add_argument('--landmark_path', type=str, default='networks/shape_predictor_68_face_landmarks.dat', help='face landmark file path')
    parser.add_argument('--parsing_path', type=str, default='networks/79999_iter.pth', help='parsing model path')
    args, _ = parser.parse_known_args()

    landmarks_detector = LandmarksDetector(args.landmark_path)
    parse_net = BiSeNet(n_classes=19)
    parse_net.cuda()
    parse_net.load_state_dict(torch.load(args.parsing_path))
    parse_net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    os.makedirs(args.output_dir, exist_ok=True)
    dst_path = os.path.join(args.output_dir, args.input_img.rsplit('/', 1)[1].split('.')[0])+'_to-'+args.project_style+'/'
    os.makedirs(dst_path, exist_ok=True)
    ori_img = cv2.imread(args.input_img)
    face_data = {'aligned_images': [], 'masks': [], 'crops': [], 'pads': [], 'quads': [], 'record_paths': []}
    print('Step1 - Face alignment and mask extraction...')
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(args.input_img), start=1):
        if i == 1:
            cv2.imwrite(dst_path + 'input.png', ori_img)
        if args.record:
            record_path = dst_path + 'face'+str(i) + '/'
            face_data['record_paths'].append(record_path)
            os.makedirs(record_path, exist_ok=True)

        # face aligned
        aligned_image, crop, pad, quad = image_align(args.input_img, face_landmarks, output_size=1024, x_scale=1, y_scale=1, em_scale=0.1)
        face_data['aligned_images'].append(aligned_image)
        face_data['crops'].append(crop)
        face_data['pads'].append(pad)
        face_data['quads'].append(quad)
        if args.record:
            aligned_image.save(record_path+'face_input.png', 'PNG')

        # mask extraction
        image_sharp = aligned_image.filter(ImageFilter.DETAIL)
        alinged_image_np = np.array(image_sharp)
        img = to_tensor(alinged_image_np)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = parse_net(img)[0]
        parsing = out.detach().squeeze(0).cpu().numpy().argmax(0)
        mask = vis_parsing_maps(alinged_image_np, parsing, stride=1)
        mask = (255 * mask).astype('uint8')
        mask = PIL.Image.fromarray(mask, 'L')
        face_data['masks'].append(mask)
        if args.record:
            mask.save(record_path+'face_mask.png', 'PNG')

    print('Step2 - Face projection and mixing back...')
    projected_images, dlatents = projector.project(face_data['aligned_images'], face_data['masks'], args.project_style)
    merged_image = ori_img
    for projected_image, dlatent, crop, quad, pad, record_path, mask in zip(projected_images, dlatents,
                                face_data['crops'], face_data['quads'], face_data['pads'], face_data['record_paths'], face_data['masks']):
        if args.record:
            projected_image.save(record_path+'face_output.png', 'PNG')
            np.save(record_path+'dlatent.npy', dlatent)
        merged_image = functions.merge_image(merged_image, projected_image, mask, crop, quad, pad)
    cv2.imwrite(dst_path+'output.png', merged_image)

if __name__ == "__main__":
    main()
