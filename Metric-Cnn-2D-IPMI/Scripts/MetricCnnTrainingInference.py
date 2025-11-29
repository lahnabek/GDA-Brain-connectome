import torch, os, argparse, sys, skimage
sys.path.append('../Packages')
import data.convert as convert
import util.tensors as tensors
import SimpleITK as sitk

from torch.utils.data import DataLoader
from tqdm import tqdm
from pde import *
from dataset import *
from model import *
from plot import *

def train(brain_id, input_dir, output_dir, gpu_device, epoch_num, learning_rate, terminating_loss, checkpoint_save_frequency):
    device = torch.device('cuda')
    torch.cuda.set_device(gpu_device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    output_dir = f'{output_dir}/{brain_id}'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model = DenseED(in_channels=2, out_channels=3, 
                    imsize=100,
                    blocks=[6, 8, 6],
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    out_activation=None,
                    upsample='nearest')
    model.train()
    model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    dataset_id = ImageDataset(input_dir, sample_name_list=[brain_id])
    dataloader_id = DataLoader(dataset_id, batch_size=1, shuffle=False, num_workers=0)

    with open(f'{output_dir}/loss.txt', 'w+') as f:
        f.write(f'From 0 to {epoch_num}\n')
        f.write(f'MSE; Adadelta: lr={learning_rate}\n')

    for epoch in tqdm(range(epoch_num)):
        epoch_loss_id = 0

        for i, batched_id_sample in enumerate(dataloader_id):
            input_id = batched_id_sample['vector_field'].to(device).float()
            input_id.requires_grad = True
            mask = batched_id_sample['mask'].float()
            u_pred_id = model(input_id)[...,1:146,1:175]
            pde_loss = pde(u_pred_id.squeeze(), input_id.squeeze(), mask.squeeze(), differential_accuracy=2)
            f_pred_id = torch.einsum('...ij,...ij->...ij', pde_loss, mask.squeeze().unsqueeze(0).expand(2,-1,-1))
            f_true_id = torch.zeros_like(f_pred_id)

            optimizer.zero_grad()
            loss_id = criterion(f_pred_id, f_true_id)
            loss_id.backward()
            epoch_loss_id += loss_id.item()
            optimizer.step()
        scheduler.step(epoch_loss_id)

        with open(f'{output_dir}/loss.txt', 'a') as f:
            f.write(f'{epoch_loss_id}\n')

        if epoch%checkpoint_save_frequency==0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_id_state_dict': optimizer.state_dict(),
            'loss_id': epoch_loss_id,
            }, f'{output_dir}/model.pth.tar')

        if epoch_loss_id<terminating_loss:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_id_state_dict': optimizer.state_dict(),
            'loss_id': epoch_loss_id,
            }, f'{output_dir}/model.pth.tar')
            break
            
    checkpoint = torch.load(f'{output_dir}/model.pth.tar')
    model = DenseED(in_channels=2, out_channels=3, 
                    imsize=100,
                    blocks=[6, 8, 6],
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    out_activation=None,
                    upsample='nearest')
    model.load_state_dict(checkpoint['model_state_dict'])

    vector_lin = convert.read_nhdr(f'{input_dir}/{brain_id}/{brain_id}_vector_field.nhdr').to(device).permute(2,0,1).float()
    mask = convert.read_nhdr(f'{input_dir}/{brain_id}/{brain_id}_filt_mask.nhdr').permute(1,0).float()
    eroded_mask = skimage.morphology.erosion(mask.cpu().numpy(), skimage.morphology.square(4))

    u_pred = model(vector_lin.unsqueeze(0))
    u_pred = u_pred[...,1:146,1:175].squeeze()
    s_pred = tensors.lin2mat(u_pred)

    metric_pred_mat = matrix_exp_2d(s_pred)
    metric_pred_lin = tensors.mat2lin(metric_pred_mat)

    file_name = f'{input_dir}/{brain_id}/{brain_id}_learned_metric_final.nhdr'
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(metric_pred_lin.cpu().detach().numpy(),(2,1,0))), file_name)
    print(f'{file_name} saved')

parser = argparse.ArgumentParser()
parser.add_argument('--brain_id', type=str, required=True, help='the HCP subject ID')
parser.add_argument('--input_dir', type=str, required=True, help='path to the brain data')
parser.add_argument('--output_dir', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--gpu_device', type=int, required=True, help='an integer for the accumulator')
parser.add_argument('--epoch_num', type=int, required=False, help='total epochs for training')
parser.add_argument('--learning_rate', type=float, required=False, help='initial learning rate of model')
parser.add_argument('--terminating_loss', type=float, required=False, help='loss threshold for termination')
parser.add_argument('--checkpoint_save_frequency', type=int, required=False, help='frequency of checkpoint save')
args = parser.parse_args()

train(brain_id=args.brain_id, input_dir=args.input_dir, output_dir=args.output_dir, gpu_device=args.gpu_device, epoch_num=args.epoch_num, learning_rate=args.learning_rate, terminating_loss=args.terminating_loss, checkpoint_save_frequency=args.checkpoint_save_frequency)