import os
import argparse
#import project_image_without_optimizer as projector   # much faster but effect worse
import project_image as projector
import numpy as np
import PIL.Image
from PIL import ImageFilter
import cv2
from tools.face_alignment import image_align
from tools.landmarks_detector import LandmarksDetector
from tools import functions

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
    args, _ = parser.parse_known_args()

    landmarks_detector = LandmarksDetector(args.landmark_path)
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
        image_sharp = aligned_image.resize((256, 256), PIL.Image.LANCZOS)
        image_sharp = image_sharp.filter(ImageFilter.DETAIL)
        alinged_image_np = np.array(image_sharp)
        mask = functions.generate_face_mask(alinged_image_np, landmarks_detector)
        mask = (255 * mask).astype('uint8')
        mask = PIL.Image.fromarray(mask, 'L')
        face_data['masks'].append(mask)
        if args.record:
            mask.save(record_path+'face_mask.png', 'PNG')

    print('Step2 - Face projection and mixing back...')
    projected_images, dlatents = projector.project(face_data['aligned_images'], face_data['masks'], args.project_style)
    merged_image = ori_img
    for projected_image, dlatent, crop, quad, pad, record_path in zip(projected_images, dlatents,
                                face_data['crops'], face_data['quads'], face_data['pads'], face_data['record_paths']):
        if args.record:
            projected_image.save(record_path+'face_output.png', 'PNG')
            np.save(record_path+'dlatent.npy', dlatent)
        merged_image = functions.merge_imge(merged_image, projected_image, crop, quad, pad)
    cv2.imwrite(dst_path+'output.png', merged_image)

if __name__ == "__main__":
    main()
