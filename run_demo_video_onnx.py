
import argparse
import onnxruntime as ort
import torch
import torchvision.transforms as standard_transforms
import cv2
import time

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def gaussian_filter_density(img_shape, points):
    '''
    Generates a density map using k-nearest neighbors for sigma calculation.

    Args:
    - img: Input image from OpenCV (can be grayscale or color).
    - points: A list of pedestrian annotations as [[col,row], [col,row], ...].

    Returns:
    - density: A density map of the same shape as the input image but with one channel.
    '''

    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)

    if gt_count == 0:
        return density

    # Build KDTree
    leafsize = 2048
    tree = KDTree(points, leafsize=leafsize)

    # Query KDTree for distances and indices of nearest neighbors
    distances, _ = tree.query(points, k=4)  # Get distances to the 4 nearest neighbors

    print('Generating density map...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        x, y = int(pt[0]), int(pt[1])  # col, row format

        if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
            pt2d[y, x] = 1.0  # Place the point on the density map

            if gt_count > 1:  # More than 1 point
                # Sum distances to the 3 nearest neighbors (skip self at distances[i][0])
                sigma = np.sum(distances[i][1:4]) * 0.1
            else:  # Single point case
                sigma = np.average(np.array(img_shape)) / 4.0  # Use average image dimension

            # Apply Gaussian filter
            density += gaussian_filter(pt2d, sigma, mode='constant')

    print('Density map generation complete.')
    return density

def process_one_image(samples, transform, session):#此函数默认batch_size为1

    ort_inputs = {'images': samples.numpy()}
    pred_logits, pred_points = session.run(['pred_logits', 'pred_points'], ort_inputs)

    outputs_scores = torch.nn.functional.softmax(torch.Tensor(pred_logits), -1)[:, :, 1][0]
    outputs_points = torch.Tensor(pred_points[0])
    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    return points, predict_cnt

def main(args):
    #读取模型
    weight_path = args.weight_path
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(weight_path, providers=providers)
    b,c,h,w = session.get_inputs()[0].shape

    #定义数据预处理的正则化部分
    transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    cap = cv2.VideoCapture(args.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter.fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t1 = time.time()
            frame = cv2.resize(frame, (w, h))
            frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = transform(frame_)
            samples = torch.Tensor(img).unsqueeze(0)
            points, count = process_one_image(samples, transform, session)

            t2 = time.time()
            print('time: {}'.format( t2 - t1), flush=True)

            # draw the predictions
            img_to_draw = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            cv2.putText(img_to_draw, 'Count: %d' % count, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

            for p in points:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

            cv2.imshow('result', img_to_draw)
            out.write(img_to_draw)

            # # Generate the density map
            # density_map = gaussian_filter_density((h, w), points)
            #
            # # Visualize the density map
            # normalized_density = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
            # normalized_density = normalized_density.astype(np.uint8)
            #
            # # Apply a colormap for better visualization
            # colored_density = cv2.applyColorMap(normalized_density, cv2.COLORMAP_JET)
            #
            # # Display the image
            # cv2.imshow("Density Map", colored_density)
            cv2.waitKey(1)

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation by onnx', add_help=False)

    parser.add_argument('--input_video', default='/home/nicola/Software/CrowdCounting-P2PNet/testData/demo.mp4',
                        help='path where to read images')
    parser.add_argument('--weight_path', default='/home/nicola/Software/CrowdCounting-P2PNet/weights/onnx/SHTechA_576x960.onnx',
                        help='path where the trained weights saved')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet_onnx evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
