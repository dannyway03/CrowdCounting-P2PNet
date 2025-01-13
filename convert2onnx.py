import argparse
import warnings
from pathlib import Path
import onnx
import onnxsim
import onnxoptimizer

from engine import *
from models import build_model

warnings.filterwarnings('ignore')


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = math.ceil(img_size / int(s)) * int(s)  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


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

    # load and check the model
    print("Simplification and optimization")
    model_onnx = onnx.load(onnx_filepath)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, 'assert ONNX model check failed'

    model_onnx = onnxoptimizer.optimize(model_onnx)
    onnx.save(model_onnx, onnx_filepath)


def main(args, debug=False):
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

    input_size_postfix = "3x" + str(args.img_size[0]) + "x" + str(args.img_size[1])
    input_batch_postfix = str(args.batch_size) + "x"

    dynamic_axes = None
    if args.batch_size < 0 or args.img_size[0] < 0 or args.img_size[1] < 0:
        print('Dynamic export is not supported')
        exit(1)

    # TODO: check why dynamic export is not working
    # if args.img_size[0] < 0 or args.img_size[1] < 0:
    #
    #     args.img_size = [576, 960]
    #     input_size_postfix = "3xHxW"  # used for output filename
    #
    #     if args.batch_size < 0:
    #         dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
    #                         'pred_logits': {0: 'batch', 1: 'points'},
    #                         'pred_points': {0: 'batch', 1: 'points'},
    #                         }
    #         args.batch_size = 1
    #         input_batch_postfix = "Nx"
    #     else:
    #         dynamic_axes = {'images': {2: 'height', 3: 'width'},  # size(1,3,640,640)
    #                         'pred_logits': {1: 'points'},
    #                         'pred_points': {1: 'points'},
    #                         }
    # else:
    #     if args.batch_size < 0:
    #         dynamic_axes = {'images': {0: 'batch'},  # size(1,3,640,640)
    #                         'pred_logits': {0: 'batch'},
    #                         'pred_points': {0: 'batch'},
    #                         }
    #         args.batch_size = 1
    #         input_batch_postfix = "Nx"

    print(
        'Creating dummy input tensor (NxCxHxW): %dx3x%dx%d' % (args.batch_size, args.img_size[0], args.img_size[1]))
    args.img_size = [check_img_size(x, 32) for x in args.img_size]
    input_tensor = torch.zeros(args.batch_size, 3, *args.img_size)  # .to(device)

    print('Setting model in eval mode')
    model.eval()
    print('Running dummy tensor inference')
    output = model(input_tensor)

    if args.onnx_filename is None:
        args.onnx_filename = Path(args.weights).stem + '_' + input_batch_postfix + input_size_postfix + '.onnx'

    f = '/'.join(['./weights/onnx', args.onnx_filename])

    model.eval()

    print('Exporting onnx')
    torch.onnx.export(model,
                      (input_tensor.cpu(),),
                      f,
                      verbose=False,
                      opset_version=13,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=['pred_logits', 'pred_points'],
                      dynamic_axes=dynamic_axes)

    # load and check the model
    print("ONNX simplification and optimization")
    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, 'assert ONNX model check failed'

    model_onnx = onnxoptimizer.optimize(model_onnx)
    onnx.save(model_onnx, f)


def get_args_parser():
    parser = argparse.ArgumentParser('ONNX export script for P2PNet', add_help=False)

    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--weights', default='./weights/SHTechA.pth',
                        help='full path of the pth weights to convert')
    parser.add_argument('--img-size', nargs='+', type=int, default=[576, 960], help='image size as H W')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--onnx-filename', default=None, type=str, help='name of onnx file to save')

    return parser


if __name__ == '__main__':
    m_parser = argparse.ArgumentParser('P2PNet Onnx Export Script', parents=[get_args_parser()])
    main(m_parser.parse_args())
