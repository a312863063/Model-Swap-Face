import PIL.Image as Image
import cv2
import numpy as np
import math

def rotate(img, degree):
    height, width = img.shape[:2]
    heightNew = round(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = round(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew))
    return imgRotation

def merge_image(bg_img, fg_img, mask, crop, quad, pad):
    bg_img_ori = bg_img.copy()
    bg_img_alpha = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
    fg_img = cv2.cvtColor(np.asarray(fg_img), cv2.COLOR_RGB2BGR)
    mask = np.asarray(mask)
    line = int(round(max(quad[2][0]-quad[0][0], quad[3][0]-quad[1][0])))
    radian = math.atan((quad[1][0]-quad[0][0])/(quad[1][1]-quad[0][1]))
    degree = math.degrees(radian)
    fg_img = rotate(fg_img, degree)
    fg_img = cv2.resize(fg_img, (line, line), interpolation=cv2.INTER_NEAREST)
    mask = rotate(mask, degree)
    mask = cv2.resize(mask, (line, line), interpolation=cv2.INTER_NEAREST)
    x1 = int(round(crop[0]-pad[0]+min([quad[0][0], quad[1][0], quad[2][0], quad[3][0]])))
    y1 = int(round(crop[1]-pad[0]+min([quad[0][1], quad[1][1], quad[2][1], quad[3][1]])))
    x2 = x1+line
    y2 = y1+line
    if x1 < 0:
        fg_img = fg_img[:, -x1:]
        mask = mask[:, -x1:]
        x1 = 0
    if y1 < 0:
        fg_img = fg_img[-y1:, :]
        mask = mask[-y1:, :]
        y1 = 0
    if x2 > bg_img.shape[1]:
        fg_img = fg_img[:, :-(x2-bg_img.shape[1])]
        mask = mask[:, :-(x2-bg_img.shape[1])]
        x2 = bg_img.shape[1]
    if y2 > bg_img.shape[0]:
        fg_img = fg_img[:-(y2 - bg_img.shape[0]), :]
        mask = mask[:-(y2 - bg_img.shape[0]), :]
        y2 = bg_img.shape[0]
    #alpha = cv2.erode(mask / 255.0, np.ones((3,3), np.uint8), iterations = 1)
    alpha = cv2.GaussianBlur(mask / 255.0, (5,5), 0)
    bg_img[y1:y2, x1:x2, 0] = (1. - alpha) * bg_img[y1:y2, x1:x2, 0] + alpha * fg_img[..., 0]
    bg_img[y1:y2, x1:x2, 1] = (1. - alpha) * bg_img[y1:y2, x1:x2, 1] + alpha * fg_img[..., 1]
    bg_img[y1:y2, x1:x2, 2] = (1. - alpha) * bg_img[y1:y2, x1:x2, 2] + alpha * fg_img[..., 2]
    bg_img[y1:y2, x1:x2] = cv2.fastNlMeansDenoisingColored(bg_img[y1:y2, x1:x2], None, 3.0, 3.0, 7, 21)

    # Seamlessly clone src into dst and put the results in output
    width, height, channels = bg_img_ori.shape
    center = (height // 2, width // 2)
    mask = 255 * np.ones(bg_img.shape, bg_img.dtype)
    normal_clone = cv2.seamlessClone(bg_img, bg_img_ori, mask, center, cv2.NORMAL_CLONE)

    return normal_clone


def generate_face_mask(im, landmarks_detector):
    from imutils import face_utils
    rects = landmarks_detector.detector(im, 1)
    # loop over the face detections
    for (j, rect) in enumerate(rects):
        """
        Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        """
        shape = landmarks_detector.shape_predictor(im, rect)
        shape = face_utils.shape_to_np(shape)

        # we extract the face
        vertices = cv2.convexHull(shape)
        mask = np.zeros(im.shape[:2],np.uint8)
        cv2.fillConvexPoly(mask, vertices, 1)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (0,0,im.shape[1],im.shape[2])
        (x,y),radius = cv2.minEnclosingCircle(vertices)
        center = (int(x), int(y))
        radius = int(radius*1.4)
        mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
        cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
        cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1)
        cv2.rectangle(mask, (0, 0), (mask.shape[1], mask.shape[0]), 0, thickness=10)
        return mask


def generate_face_mask_without_hair(im, landmarks_detector, ie_polys=None):
    # get the mask of the image with only face area
    rects = landmarks_detector.detector(im, 1)
    image_landmarks = np.matrix([[p.x, p.y] for p in landmarks_detector.shape_predictor(im, rects[0]).parts()])
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    # hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(im.shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    hull_mask = hull_mask.squeeze()
    return hull_mask
