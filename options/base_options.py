import argparse
from pathlib import Path
# 【修改点 1】: 改为从你的 utils 文件夹导入 tools
from utils import tools


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument("--dataroot", default="./data", help="path to images")
        parser.add_argument("--name", type=str, default="experiment_name",
                            help="name of the experiment. It decides where to store samples and models")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
        # model parameters
        parser.add_argument("--input_nc", type=int, default=3,
                            help="# of input image channels: 3 for RGB and 1 for grayscale")
        parser.add_argument("--output_nc", type=int, default=3,
                            help="# of output image channels: 3 for RGB and 1 for grayscale")
        # dataset parameters
        parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
        parser.add_argument("--img_size", type=int, default=128,
                            help="square training and inference resolution")
        parser.add_argument("--load_size", type=int, default=286, help="scale images to this size")
        parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
        # additional parameters
        parser.add_argument("--epoch", type=str, default="latest",
                            help="which epoch to load? set to latest to use latest cached model")
        parser.add_argument("--suffix", default="", type=str, help="customized suffix: opt.name = opt.name + suffix")

        self.initialized = True
        return parser

    def gather_options(self):
        """
        【修改点 2】: 删除了原版 CycleGAN 中复杂的 models 和 data 的动态挂载逻辑，
        直接解析基础参数，防止出现 Attribute 相关的报错。
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # 保存 parser 以备打印时读取默认值
        self.parser = parser
        opt, _ = parser.parse_known_args()
        return opt

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        # 【修改点 3】: 调用 tools.mkdirs 而不是 util.mkdirs
        tools.mkdirs(expr_dir)
        file_name = expr_dir / f"{opt.phase}_opt.txt"
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        self.print_options(opt)
        self.opt = opt
        return self.opt
