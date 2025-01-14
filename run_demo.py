import argparse
import glob
import os
import time
import warnings

from PIL import Image

from engine import *
from models import build_model

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--weights', default='/home/nicola/Software/CrowdCounting-P2PNet/weights/SHTechA.pth',
                        help='path .pth weights')
    parser.add_argument('--input_video', default='/home/nicola/Software/CrowdCounting-P2PNet/testData/demo.mp4',
                        help='path to input video')
    return parser


def Predict(model, img_path, transform, device):
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    # draw the predictions
    size = 4
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

    cv2.putText(img_to_draw, f'Predict Crowd Count: {predict_cnt}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2)  # 在图片上写文字
    # save the visualized image
    # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)
    return img_to_draw


def process_frame(samples, model):
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    return points, predict_cnt


def main(args):
    print('Building model:')
    print('   backbone: %s' % args.backbone)
    print('   anchor cols x rows: %s x %s' % (args.line, args.row))
    model = build_model(args).to('cpu')

    print('Loading model weights: %s' % args.weights)
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        print('Loading checkpoint failed')
        exit(1)

    print('Setting model in eval mode')
    model.eval()

    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    w, h = 1280, 768

    cap = cv2.VideoCapture(args.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = width // 128 * 128
    height = height // 128 * 128

    w = min(w, width)
    h = min(h, height)

    video_out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter.fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t1 = time.time()
            frame = cv2.resize(frame, (w, h))
            frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = transform(frame_)
            samples = torch.Tensor(img).unsqueeze(0)
            points, count = process_frame(samples, model)

            t2 = time.time()
            print('time: {}'.format(t2 - t1), flush=True)

            # draw the predictions
            img_to_draw = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            cv2.putText(img_to_draw, 'Count: %d' % count, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

            for p in points:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

            cv2.imshow('result', img_to_draw)
            video_out.write(img_to_draw)
            cv2.waitKey(1)

    video_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    main(parser.parse_args())

# import cv2
# import matplotlib.pyplot as plt
#
# # 读取原始图片
# img = cv2.imread('/mnt/d/MyDocs/Datasets/mall_dataset/frames/seq_000007.jpg')
#
# # 对图片进行处理
# processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 创建一个 1x2 的图像网格，左侧显示原始图片，右侧显示处理后的图片
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# # 在左侧显示原始图片
# ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# ax1.set_title('Original Image')
#
# # 在右侧显示处理后的图片
# ax2.imshow(processed_img, cmap='gray')
# ax2.set_title('Processed Image')
#
# # 隐藏坐标轴
# ax1.axis('off')
# ax2.axis('off')
#
# # 显示图像
# plt.show()
