from model_training.network import str2Model
import argparse
import torch
import math
from thop.profile import profile
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from torch.nn import Softmax
from metrics import Metrics
import pickle
import copy

IMG = "JPEGImages"
LABLEL = "Annotations"

last_file = {
    'CAU7': 38,
    'CRA17': 61,
    'CRA27': 38,
    'CAU31': 49,
    'CRA31': 53,
    'CRA34': 50,
    'CAU26': 43,
    'CRA0': 52,
    'CRA9': 55,
    'CRA23': 56
}

def sort_file(files):
    """
    Sorts a list of file paths based on the length of the file names.
    
    Args:
        files (list): A list of file paths.
        
    Returns:
        list: A sorted list of file paths.
    """
    files = list(map(lambda x: (len(x.split('/')[-1]), x), files))
    files.sort()
    return list(map(lambda x: x[1], files))


def get_files(path):
    """
    Get a list of files in the specified path, excluding '.DS_Store' if present.

    Args:
        path (str): The path to the directory.

    Returns:
        list: A list of file names in the directory.
    """
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    return files

def to_label(x):
    """
    Converts a tensor to a label array.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        numpy.ndarray: The converted label array.
    """
    x = x.cpu().detach().numpy()
    x = x.astype('uint8')
    return x

from PIL import Image

def save_img(x, path):
    """
    Save an image to the specified path.
    
    Args:
        x (numpy.ndarray): The image data as a NumPy array.
        path (str): The path where the image should be saved.
    """
    img = Image.fromarray(x)
    img.save(path)

def get_coord(low, high):
    """
    Calculate the coordinates based on the given low and high values.
    
    Parameters:
    low (int): The lower bound value.
    high (int): The upper bound value.
    
    Returns:
    tuple: A tuple containing the calculated coordinates in the format (low, mid_low, mid_high, high).

    get_coord(0, 5) -> 0, 2, 3, 5 means 0~1, 2, 3~4
    """
    span = high - low
    span = span // 2
    return low, low + span, high - span, high

def bound_the_box(img, left, right, top, bottom):
    """
    Adjusts the boundaries of a box based on the first non-zero pixel in each direction.

    Args:
        img (numpy.ndarray): The image array.
        left (int): The left boundary of the box.
        right (int): The right boundary of the box.
        top (int): The top boundary of the box.
        bottom (int): The bottom boundary of the box.

    Returns:
        tuple: A tuple containing the adjusted left, right, top, and bottom boundaries of the box.
    """
    new_left, new_right, new_top, new_bottom = left, right, top, bottom
    for i in range(left, right):
        if img[top:bottom, i].sum() > 0:
            new_left = i
            break
    for i in range(right-1, left-1, -1):
        if img[top:bottom, i].sum() > 0:
            new_right = i+1
            break
    for i in range(top, bottom):
        if img[i, left:right].sum() > 0:
            new_top = i
            break
    for i in range(bottom-1, top-1, -1):
        if img[i, left:right].sum() > 0:
            new_bottom = i+1
            break
    return new_left, new_right, new_top, new_bottom

def preprocess(img):
    """
    Preprocesses an image by removing small connected components.
    
    Args:
        img (numpy.ndarray): The input image.
        
    Returns:
        numpy.ndarray: The preprocessed image.
    """
    n, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    for i in range(1, n):
        size = stats[i, cv2.CC_STAT_AREA]
        if size < 5:
            img[output == i] = 0
    return img

def adaptive_patch(img, length_threshold, threshold, one_patch=False):
    """
    Divides an image into adaptive patches based on certain criteria.
    Once one_patch is set to True, the function will only return one patch,
    and the other two parameters are meaningless

    Args:
        img (numpy.ndarray): The input image.

        length_threshold (float): The threshold for determining the minimum length of a patch.
            The actual minimum length is calculated as length_threshold * (img.shape[0] / 512).
            If one_patch is True, this parameter is ignored and the minimum length is set to 1024.

        threshold (float): The threshold for determining whether a patch should be further divided.
            The value is compared against the ratio of the area of the patch to its width and height.
            If the ratio is greater than threshold, the patch is not divided further.

        one_patch (bool, optional): If True, the function will only return one patch.
            Defaults to False.

    Returns:
        list: A list of tuples representing the coordinates of the adaptive patches.

    Raises:
        ValueError: If the input image dimensions are not valid.

    """

    list_of_box = []
    MIN_LENGTH = length_threshold * (img.shape[0] / 512)
    if one_patch:
        MIN_LENGTH = 1024
    THRESHOLD = threshold
    img = preprocess(img)

    def cal_box(img, left=None, right=None, top=None, bottom=None):
        """
        Recursive function to calculate the adaptive patches.

        Args:
            img (numpy.ndarray): The input image.
            left (int, optional): The left coordinate of the current box. Defaults to None.
            right (int, optional): The right coordinate of the current box. Defaults to None.
            top (int, optional): The top coordinate of the current box. Defaults to None.
            bottom (int, optional): The bottom coordinate of the current box. Defaults to None.

        Raises:
            ValueError: If the input box coordinates are not valid.

        """

        height, width = img.shape
        if left is None and right is None and top is None and bottom is None:
            left, right, top, bottom = 0, width, 0, height
        elif left is None or right is None or top is None or bottom is None:
            raise ValueError('left, right, top, bottom must be all None or all not None')

        left, right, top, bottom = bound_the_box(img, left, right, top, bottom)

        area = 0
        mode = 0

        if bottom - top > MIN_LENGTH*2:
            top_start, top_end, bottom_start, bottom_end = get_coord(top, bottom)
            left1, right1, top1, bottom1 = bound_the_box(img, left, right, top_start, top_end)
            left2, right2, top2, bottom2 = bound_the_box(img, left, right, bottom_start, bottom_end)
            area = (right1 - left1) * (bottom1 - top1) + (right2 - left2) * (bottom2 - top2)
            mode = 1

        if right - left > MIN_LENGTH*2:
            left_start, left_end, right_start, right_end = get_coord(left, right)
            left1, right1, top1, bottom1 = bound_the_box(img, left_start, left_end, top, bottom)
            left2, right2, top2, bottom2 = bound_the_box(img, right_start, right_end, top, bottom)
            area_horizontal = (right1 - left1) * (bottom1 - top1) + (right2 - left2) * (bottom2 - top2)
            if area_horizontal < area:
                area = area_horizontal
                mode = 2
            

        if mode == 0 or area / (right - left) / (bottom - top) > THRESHOLD:
            list_of_box.append((left, right, top, bottom))
        elif mode == 1:
            top_start, top_end, bottom_start, bottom_end = get_coord(top, bottom)
            if (top - bottom) % 2 == 1:
                top_end += 1
            cal_box(img, left, right, top_start, top_end)
            cal_box(img, left, right, bottom_start, bottom_end)
        else:
            left_start, left_end, right_start, right_end = get_coord(left, right)
            if (left - right) % 2 == 1:
                left_end += 1
            cal_box(img, left_start, left_end, top, bottom)
            cal_box(img, right_start, right_end, top, bottom)
    
    img2 = img.copy().astype(np.int32)
    cal_box(img2)
    return list_of_box

def get_label(label_path, index):
    label = get_files(label_path)
    label = sort_file(label)
    img = Image.open(os.path.join(label_path, label[index])).convert('L')
    img = np.array(img)
    return img

def box_post_process(margin, img_size):
    def function(box):
        ratio = 512 / img_size
        left, right, top, bottom = box
        left, top, right, bottom = left * ratio, top * ratio, right * ratio, bottom * ratio
        top = math.floor(top / 16) * 16
        left = math.floor(left / 16) * 16
        bottom = math.ceil(bottom / 16) * 16
        right = math.ceil(right / 16) * 16
        if left > margin and right < 512 - margin:
            left -= margin
            right += margin
        if top > margin and bottom < 512 - margin:
            top -= margin
            bottom += margin

        return (left, right, top, bottom)
    return function

def post_process(img):
    img = to_label(img)
    n, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    for i in range(1, n):
        size = stats[i, cv2.CC_STAT_AREA]
        if size < 10:
            img[output == i] = 0
    return torch.tensor(img)

def segment_boxes(img, boxes, model):
    softmax = Softmax(dim=1)
    ops = 0
    prob_map = torch.zeros_like(img[0]).cpu()
    count_map = torch.zeros_like(img[0]).cpu()
    img = img.cuda()
    for box in boxes:
        left, right, top, bottom = box
        img2 = img[:, top:bottom, left:right].unsqueeze(0)
        dic = model(img2)
        ops_temp, _ = profile(model, inputs=(img2.cuda(), ), verbose=False)
        ops += ops_temp
        prob = softmax(dic[1])[0][1]
        prob_map[top:bottom, left:right] += prob.clone().detach().cpu()
        count_map[top:bottom, left:right] += 1

    prob_map[count_map > 0] /= count_map[count_map > 0]
    prob_map[prob_map > 0.5] = 1
    prob_map[prob_map <= 0.5] = 0

    return ops, prob_map

def draw_boxes(img, boxes, label, no_bounding_box=False, no_gt=True):
    draw = np.stack([img, img, img], axis=2)
    draw *= 255
    if not no_bounding_box:
        if not no_gt:
            draw[:,:,0][(draw[:,:,0]==0) & (label==255)] = 255
            draw[:,:,0][(draw[:,:,0]==255) & (label==0)] = 0
            draw[:,:,1][(draw[:,:,0]==255) & (label==0)] = 0
        
        for box in boxes:
            left, right, top, bottom = box
            blk = np.zeros_like(draw, dtype=np.uint8)
            cv2.rectangle(blk, (left, top), (right, bottom), (0, 255, 0), -1)
            draw = cv2.addWeighted(draw, 1, blk, 0.2, 0)
    return draw

def adjust(label, patch_splitting):
    label = to_label(label)
    adjusted_boxes = patch_splitting(label)
    return adjusted_boxes

def load_img(transform, image_path, first_img_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    fimg = Image.open(first_img_path).convert('RGB')
    fimg = transform(fimg)
    img[0] = fimg[0]
    img.unsqueeze_(0)

    return img

def cover_rate(label, boxes):
    label = to_label(label)
    label = label.astype(np.int32)
    original_sum = np.sum(label)
    for box in boxes:
        left, right, top, bottom = box
        label[top:bottom, left:right] = 0
    if original_sum == 0:
        return 0
    return 1 - np.sum(label) / original_sum

def get_transforms(IMGSIZE):
    load_transform = transforms.Compose([
        transforms.Resize((IMGSIZE, IMGSIZE)),
        transforms.ToTensor(),
    ])
    regular_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return load_transform, regular_transform

# todo change to video_runner
def video_runner(args, model, IMG_SIZE, MARGIN, patch_splitting):
    assert args.pixel_thresh != None
    assert args.cont != None
    assert args.video_thresh != None
    load_transform, regular_transform = get_transforms(IMG_SIZE)
    eval_path = '/home/charlieyao/XMem_total/eval_dataset/' + args.path
    videos = get_files(os.path.join(eval_path, IMG))

    total_ops = [0, 0]
    pred, gt, video_boxes, logs = {}, {}, {}, {}
    
    for video in videos:
        video_dir = os.path.join(eval_path, IMG, video)
        label_path = os.path.join(eval_path, LABLEL, video)
        images = get_files(video_dir)
        images = sort_file(images)
        first_img = images[0]
        pred[video] = []
        gt[video] = []
        video_boxes[video] = []
        last_boxes = []
        steady = False
    
        count = 0
        for index, image in enumerate(images):
            if video[:4] != 'CVAI' and index == last_file[video] - 10 + 1:
                break
            image_path = os.path.join(video_dir, image)
            first_img_path = os.path.join(video_dir, first_img)
            label = get_label(label_path, index)
            stage1_box = None
            if not steady:
                stage1_box = (IMG_SIZE, IMG_SIZE, IMG_SIZE, IMG_SIZE)

                img = load_img(load_transform, image_path, first_img_path)
                img = img.cuda()

                _, x = model(img)
                x = torch.argmax(x, dim=1)[0]
                x = x.cpu().detach().numpy()
                x = x.astype('uint8')
                ops, _ = profile(model, inputs=(img, ), verbose=False)
                total_ops[0] += ops
                pixel_sum = int(np.sum(x))
                
                boxes = patch_splitting(x)
                boxes = list(map(box_post_process(MARGIN, IMG_SIZE), boxes))

            # if the region is too small, we will not segment it
            if pixel_sum > 850 * ((IMG_SIZE / 512) ** 2):
                img = load_img(regular_transform, image_path, first_img_path)
                ops, prob_map = segment_boxes(img[0], boxes, model)
                prob_map = post_process(prob_map)
                total_ops[1] += ops

                if steady:
                    boxes = adjust(prob_map, patch_splitting)
                    boxes = list(map(box_post_process(MARGIN, 512), boxes))
                else:
                    # check if turning into steady mode
                    if (cover_rate(prob_map, last_boxes) > args.video_thresh) and (prob_map.sum().item() > args.pixel_thresh):
                        last_box_label = torch.zeros_like(prob_map)
                        for box in last_boxes:
                            left, right, top, bottom = box
                            last_box_label[top:bottom, left:right] = prob_map[top:bottom, left:right]
                        last_boxes = adjust(last_box_label, patch_splitting)
                        last_boxes = list(map(box_post_process(MARGIN, 512), last_boxes))
                        count += 1
                    else:
                        last_boxes = boxes
                        count = 0
                    if count > args.cont:
                        steady = True
                        boxes = adjust(prob_map, patch_splitting)
                        boxes = list(map(box_post_process(MARGIN, 512), boxes))
            else:
                prob_map = torch.zeros((512, 512), dtype=torch.uint8)

            pred[video].append(prob_map)
            gt[video].append(label)
            boxes_copy = copy.deepcopy(boxes)
            if stage1_box != None:
                boxes_copy.insert(0, stage1_box)
            video_boxes[video].append(boxes_copy)
        
    return pred, gt, tuple(total_ops), video_boxes, logs

def adaptive_runner(args, model, IMG_SIZE, MARGIN, one_patch=False):
    assert MARGIN != None
    assert IMG_SIZE != 512
    assert args.label_thresh != None
    assert args.area_thresh != None
    def patch_splitting(x):
        return adaptive_patch(x, args.label_thresh, args.area_thresh)

    return video_runner(args, model, IMG_SIZE, MARGIN, patch_splitting)

def one_patch_runner(args, model, IMG_SIZE, MARGIN):
    def patch_splitting(x):
        return adaptive_patch(x, args.label_thresh, args.area_thresh, one_patch=True)

    return video_runner(args, model, IMG_SIZE, MARGIN, patch_splitting)

def plain_runner(args, model, IMG_SIZE, MARGIN):
    _, regular_transform = get_transforms(IMG_SIZE)
    eval_path = '/home/charlieyao/XMem_total/eval_dataset/' + args.path
    videos = get_files(os.path.join(eval_path, IMG))

    total_ops = 0
    pred, gt, video_boxes, logs = {}, {}, {}, {}
    
    for video in videos:
        video_dir = os.path.join(eval_path, IMG, video)
        label_path = os.path.join(eval_path, LABLEL, video)
        images = get_files(video_dir)
        images = sort_file(images)
        first_img = images[0]
        pred[video] = []
        gt[video] = []
        video_boxes[video] = []
        for index, image in enumerate(images):
            # to be able to just run the whole video
            if video[:4] != 'CVAI' and index == last_file[video] - 10 + 1:
                break
            image_path = os.path.join(video_dir, image)
            first_img_path = os.path.join(video_dir, first_img)
            label = get_label(label_path, index)

            img = load_img(regular_transform, image_path, first_img_path)
            img = img.cuda()

            boxes = [(0, 512, 0, 512)]
            ops, prob_map = segment_boxes(img[0], boxes, model)
            total_ops += ops

            pred[video].append(prob_map)
            gt[video].append(label)
            video_boxes[video].append(boxes)

    return pred, gt, (0, total_ops), video_boxes, logs

def single_runner(args, model, IMG_SIZE, MARGIN, patch_splitting):
    load_transform, regular_transform = get_transforms(IMG_SIZE)
    eval_path = '/home/charlieyao/XMem_total/eval_dataset/' + args.path
    videos = get_files(os.path.join(eval_path, IMG))

    total_ops = [0, 0]
    pred, gt, video_boxes, logs = {}, {}, {}, {}

    for video in videos:
        video_dir = os.path.join(eval_path, IMG, video)
        label_path = os.path.join(eval_path, LABLEL, video)
        images = get_files(video_dir)
        images = sort_file(images)
        first_img = images[0]
        pred[video] = []
        gt[video] = []
        video_boxes[video] = []
        for index, image in enumerate(images):
            if video[:4] != 'CVAI' and index == last_file[video] - 10 + 1:
                break

            image_path = os.path.join(video_dir, image)
            first_img_path = os.path.join(video_dir, first_img)
            img = load_img(load_transform, image_path, first_img_path)
            label = get_label(label_path, index)
            stage1_box = (IMG_SIZE, IMG_SIZE, IMG_SIZE, IMG_SIZE)
            boxes = []

            img = img.cuda()
            _, x = model(img)
            x = torch.argmax(x, dim=1)[0]
            x = x.cpu().detach().numpy()
            x = x.astype('uint8')
            ops, _ = profile(model, inputs=(img, ), verbose=False)
            total_ops[0] += ops
            
            if x.sum() > 850 * ((IMG_SIZE / 512) ** 2):
                boxes = patch_splitting(x)
                img = load_img(regular_transform, image_path, first_img_path)
                ops, prob_map = segment_boxes(img[0], boxes, model)
                total_ops[1] += ops
            else:
                prob_map = torch.zeros((512, 512), dtype=torch.uint8)
            if video == 'CRA9' and index == 20:
                l = None

            pred[video].append(prob_map)
            gt[video].append(label)
            boxes.insert(0, stage1_box)
            video_boxes[video].append(boxes)
        
    return pred, gt, tuple(total_ops), video_boxes, logs

def one_patch_single_runner(args, model, IMG_SIZE, MARGIN):
    def patch_splitting(img):
        boxes = adaptive_patch(img, 0, 0, one_patch=True)
        boxes = list(map(box_post_process(MARGIN, IMG_SIZE), boxes))
        return boxes
    return single_runner(args, model, IMG_SIZE, MARGIN, patch_splitting)

def adaptive_single_runner(args, model, IMG_SIZE, MARGIN):
    assert MARGIN != None
    assert IMG_SIZE != 512
    assert args.label_thresh != None
    assert args.area_thresh != None
    def patch_splitting(img):
        boxes = adaptive_patch(img, args.label_thresh, args.area_thresh)
        boxes = list(map(box_post_process(MARGIN, IMG_SIZE), boxes))

        return boxes
    return single_runner(args, model, IMG_SIZE, MARGIN, patch_splitting)

def tiled_single_runner(args, model, IMG_SIZE, MARGIN):
    assert MARGIN != None
    assert IMG_SIZE != 512
    assert args.label_thresh != None
    def tiled_patch(img):
        """
        Extracts tiled patches from an image.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            list: A list of tuples representing the coordinates of each tiled patch.

        """
        list_of_box = []
        length = img.shape[0]
        size = int(args.label_thresh * (length / 512))

        height, width = img.shape[0], img.shape[1]
        _, _, top, bottom = bound_the_box(img, 0, width, 0, height)

        i = bottom
        while i - size >= top:
            row = img[i-size:i, :]
            left, right, _, _ = bound_the_box(row, 0, row.shape[1], 0, row.shape[0])

            j = left
            while j + size <= right:
                list_of_box.append((j, j + size, i - size, i))
                j += size
            if j < right:
                l = max(0, right - size)
                list_of_box.append((l, right, i - size, i))
            i -= size
        if i > top:
            bot = min(top+size, img.shape[0])
            row = img[top:bot, :]
            left, right, _, _ = bound_the_box(row, 0, row.shape[1], 0, row.shape[0])

            j = left
            while j + size <= right:
                list_of_box.append((j, j + size, top, bot))
                j += size
            if j < right:
                l = max(0, right - size)
                list_of_box.append((l, right, top, bot))
        return list(map(box_post_process(MARGIN, IMG_SIZE), list_of_box))

    
    return single_runner(args, model, IMG_SIZE, MARGIN, tiled_patch)

def get_runner(method):
    runners = {
        'adaptive': adaptive_runner,
        'tiled_single': tiled_single_runner,
        'one_patch': one_patch_runner,
        'plain': plain_runner,
        'one_patch_single': one_patch_single_runner,
        'adaptive_single': adaptive_single_runner
    }
    return runners[method]

def run():
    args = argparse.ArgumentParser()
    args.add_argument('-mo', '--model', type=str, required=True)
    args.add_argument('-me', '--method', type=str, required=True)
    args.add_argument('-pa', '--path', type=str, required=True)
    args.add_argument('-com', '--comment', type=str, default='')
    args.add_argument('-e', '--epoch', type=int, default=60000)
    args.add_argument('-s', '--size', type=int, default=512)
    args.add_argument('-ma', '--margin', type=int, default=None)
    args.add_argument('-lt', '--label_thresh', type=int, default=None)
    args.add_argument('-at', '--area_thresh', type=float, default=None)
    args.add_argument('-c', '--cont', type=int, default=None)
    args.add_argument('-pi', '--pixel_thresh', type=int, default=None)
    args.add_argument('-vt', '--video_thresh', type=float, default=None)
    args.add_argument('--output', type=str, default=None)
    args = args.parse_args()

    # single means no video involved
    methods = ['adaptive', 'tiled_single', 'one_patch', 'plain', 'one_patch_single', 'adaptive_single']

    # the pixel threshold is expressed as the sum of RGB channels
    if args.pixel_thresh != None and args.pixel_thresh % 3 != 0:
        raise ValueError('Invalid pixel_thresh')
    args.pixel_thresh = args.pixel_thresh // 3

    if args.method not in methods:
        raise ValueError('Invalid method')
    if '/' in args.path:
        raise ValueError('Invalid path')
    
    model = torch.load('output/' + args.model + f'/checkpoint_{args.epoch}.pth')
    model2 = str2Model(model['model_type'])()
    model = model['network']
    model2.load_state_dict(model)
    model = model2.cuda().eval()
    
    # generate output_path name
    output_path = args.output
    if output_path is None:
        comment = args.comment + '_' if args.comment else ''
        output_path = f'output/{args.model}/{args.method}_{args.epoch}/{comment}{args.size}_{args.margin}_{args.label_thresh}_{args.area_thresh}_{args.cont}_{args.video_thresh}_{args.pixel_thresh*3}'
    print(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    else:
        raise ValueError('Output path already exists')

    IMG_SIZE = args.size
    MARGIN = args.margin

    if IMG_SIZE % 16 != 0:
        raise ValueError('Size must be a multiple of 16')

    runner = get_runner(args.method)
    pred, gt, ops, boxes, logs = runner(args, model, IMG_SIZE, MARGIN)

    # log should contain log infos
    for log in logs:
        with open(os.path.join(output_path, log), 'w') as f:
            f.write(logs[log])

    # do the evaluation and save the results
    metric = Metrics()
    for video in pred:
        os.makedirs(os.path.join(output_path, video), exist_ok=True)
        for index, img in enumerate(pred[video]):
            # save_segmentation result
            img = img.cpu().detach().numpy()
            img = img.astype(np.uint8)
            img = img * 255
            save_img(img, os.path.join(output_path, f'{video}/{index}.png'))

            # save difference result
            gt_img = gt[video][index]
            draw = np.stack([img, img, img], axis=2)
            draw2 = draw.copy()
            mask = (draw2[:,:,0]==0) & (gt_img==255)
            draw2[mask] = [255, 0, 0]
            mask = (draw2[:,:,0]==255) & (gt_img==0)
            draw2[mask] = [0, 255, 0]
            os.makedirs(os.path.join(output_path+'/diff', video), exist_ok=True)
            save_img(draw2, os.path.join(output_path+'/diff', f'{video}/{index}.png'))
            
            # save bounding box results
            draw3 = (img.copy() / 255).astype(np.uint8)
            box_required = boxes[video][index]
            l = box_required[0][0]
            equal = True
            for i in range(1, 4):
                if box_required[0][i] != l:
                    equal = False
                    break
            # the first is for Stage 1 time profiling purpose
            if equal:
                box_required = box_required[1:]
            draw3 = draw_boxes(draw3, box_required, gt_img)
            os.makedirs(os.path.join(output_path+'/boxes', video), exist_ok=True)
            save_img(draw3, os.path.join(output_path+'/boxes', f'{video}/{index}.png'))

            metric.update_np(img, gt_img)

    # dump video_boxes pickle
    with open(os.path.join(output_path, 'video_boxes.pkl'), 'wb') as f:
        pickle.dump(boxes, f)
    with open(os.path.join(output_path, 'evaluation.txt'), 'w') as f:
        f.write(str(metric))
        f.write(f'\nTotal ops: {ops}')
    metric.print()

if __name__ == '__main__':
    run()