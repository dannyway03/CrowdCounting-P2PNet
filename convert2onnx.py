import argparse
import warnings
from pathlib import Path

import onnx
import onnxsim

from engine import *
from models import build_model

warnings.filterwarnings('ignore')


def export_onnx(model, input_tensor, onnx_filepath, opset_version=13, dynamic=False):
    torch.onnx.export(
        model,
        input_tensor,
        onnx_filepath,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['pred_logits', 'pred_points'],
        dynamic_axes={  # review this part
            'images': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1, 3, 1280, 768)

            'pred_logits': {
                0: 'batch',
                1: 'points'},

            'pred_points': {
                0: 'batch',
                1: 'points'}
        } if dynamic else None)

    model_onnx = onnx.load(onnx_filepath)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(model_onnx)

    assert check, 'assert ONNX model check failed'
    onnx.save(model_onnx, onnx_filepath)


def main(args, debug=False):
    device = torch.device('cpu')
    model = build_model(args).to(device)

    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()

    input_tensor = torch.rand((1, 3, 576, 960), dtype=torch.float32).to(device)
    out = model(input_tensor)

    if args.onnx_filename is None:
        args.onnx_filename = Path(args.weights).stem + '_576x960.onnx'

    export_onnx(model, input_tensor, '/'.join(['./weights/onnx', args.onnx_filename]), opset_version=13, dynamic=False)


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--weights', default='./weights/SHTechA.pth',
                        help='full path of the pth weights to convert')
    parser.add_argument('--onnx_filename', default=None, type=str, help='name of onnx file to save')

    return parser


if __name__ == '__main__':
    m_parser = argparse.ArgumentParser('P2PNet Onnx Export Script', parents=[get_args_parser()])
    main(m_parser.parse_args())
