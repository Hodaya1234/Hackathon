import torch.utils.data as data_utils
import torch

class DataSet(data_utils.Dataset):
    def __init__(self, x, y):
        super(DataSet, self).__init__()
        self.all_x = torch.from_numpy(x)
        self.all_y = torch.from_numpy(y)
        self.all_y = self.all_y.long()
        self.n_data = len(y)

    def __getitem__(self, index):
        x = self.all_x[index]
        y = self.all_y[index]
        return x,y

    def __len__(self):
        return len(self.all_y)



