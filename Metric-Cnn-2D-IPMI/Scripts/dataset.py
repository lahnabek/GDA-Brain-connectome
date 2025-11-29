import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from skimage import filters


class ImageDataset(Dataset):
    def __init__(self, data_dir, sample_name_list):
        self.data_dir = data_dir
        self.sample_name_list = sample_name_list
        self.sample_name_list.sort()
        
    def __len__(self):
        return len(self.sample_name_list)
        
    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sample_name = self.sample_name_list[idx]
        vector_field_path = f'{self.data_dir}/{sample_name}/{sample_name}_vector_field.nhdr'
        mask_path = f'{self.data_dir}/{sample_name}/{sample_name}_filt_mask.nhdr'

        vector_field = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(vector_field_path))).permute(2,0,1).to(device)*1000.0
        mask = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(mask_path))).permute(1,0)
        boundary_mask = torch.where(torch.from_numpy(filters.laplace(mask))>0,1,0)
        mask = (mask-boundary_mask).to(device)

        sample = {  'vector_field'  : vector_field,
                    'mask'          : mask.unsqueeze(0)}
        return sample
    