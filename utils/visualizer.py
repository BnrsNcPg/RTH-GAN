# 文件位置：D:\shiyan\PINN-Model\utils\visualizer.py

import os
import time
from . import tools  # 【修改这里】：从同级目录导入你自己的 tools.py

class Visualizer():
    """
    负责在训练过程中保存图片和打印日志的记录类
    """
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.savedir = os.path.join(opt.checkpoints_dir, opt.name)
        self.img_dir = os.path.join(self.savedir, 'images')
        self.log_name = os.path.join(self.savedir, 'loss_log.txt')

        # 【修改这里】：调用 tools 里的方法
        tools.mkdirs([self.savedir, self.img_dir])

        # 初始化日志文件
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f'================ Training Loss Log ({now}) ================\n')

    def display_current_results(self, visuals, epoch, save_result):
        if save_result:
            for label, image in visuals.items():
                # 【修改这里】：调用 tools 里的方法
                image_numpy = tools.tensor2im(image)
                img_path = os.path.join(self.img_dir, f'epoch{epoch}_{label}.png')
                tools.save_image(image_numpy, img_path)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = f'(epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) '
        for k, v in losses.items():
            message += f'{k}: {v:.3f} '

        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write(f'{message}\n')