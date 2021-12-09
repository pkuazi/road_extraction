import os
import torch
# from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# from dataloaders.utils import decode_seg_map_sequence

# def decode_segmap(label_mask, dataset, plot=False):
#     """Decode segmentation class labels into a color image
#     Args:
#         label_mask (np.ndarray): an (M,N) array of integer values denoting
#           the class label at each spatial location.
#         plot (bool, optional): whether to show the resulting color image
#           in a figure.
#     Returns:
#         (np.ndarray, optional): the resulting decoded color image.
#     """
    
#     n_classes = 10
#     label_colours = np.array([
#     [204, 204, 255],    # 卷云Ci  1
#     [102, 102, 255],    # 卷层云Sc  2
#     [0, 0, 255],    # 深对流云Dc  3
#     [255, 255, 204],    # 高积云Ac  4
#     [255, 255, 0],    # 高层云As  5
#     [204, 204, 0],    # 雨层云Ns  6
#     [255, 153, 153],    # 积云Cu  7
#     [255, 102, 51],    # 层积云Sc  8
#     [255, 0, 0],    # 层云St  9
#     [255, 255, 255]    # 其他  255
#     ])

#     r = label_mask.copy()
#     g = label_mask.copy()
#     b = label_mask.copy()
#     for ll in range(0, n_classes):
#         r[label_mask == ll] = label_colours[ll, 0]
#         g[label_mask == ll] = label_colours[ll, 1]
#         b[label_mask == ll] = label_colours[ll, 2]
#     rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
#     rgb[:, :, 0] = r / 255.0
#     rgb[:, :, 1] = g / 255.0
#     rgb[:, :, 2] = b / 255.0
#     if plot:
#         plt.imshow(rgb)
#         plt.show()
#     else:
#         return rgb

# def decode_seg_map_sequence(label_masks, dataset='pascal'):
#     rgb_masks = []
#     for label_mask in label_masks:
#         rgb_mask = decode_segmap(label_mask, dataset)
#         rgb_masks.append(rgb_mask)
#     rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
#     return rgb_masks

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))    # define a SummaryWriter() instance
        return writer

    # def visualize_image(self, writer, dataset, image, target, output, global_step):
    #     grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
    #     writer.add_image('Image', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     writer.add_image('Predicted label', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     # torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度
    #     # torch.tensor.detach():return a new Variable,which is separated from the current calculation graph,but still points to
    #     #                       the original variable.The difference is that requires_grad is false.
    #     writer.add_image('Groundtruth label', grid_image, global_step)