import torch
from torch.utils.datasets import Dataset

# PyTorch dataset retrieves the dataset’s features and labels one sample at a time
# Create a custom Dataset class for the reviews
class ReviewDataset(Dataset):
    
    def __init__(self, input_ids_list, label_id_list):
        self.input_ids_list = input_ids_list
        self.label_id_list = label_id_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, item):
        # convert list of token_ids into an array of PyTorch LongTensors
        input_ids = json.loads(self.input_ids_list[item]) 
        label_id = self.label_id_list[item]

        input_ids_tensor = torch.LongTensor(input_ids)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)

        return input_ids_tensor, label_id_tensor

# PyTorch DataLoader helps to to organise the input training data in “minibatches” and reshuffle the data at every epoch
# It takes Dataset as an input
def create_data_loader(path, batch_size): 
    print("Get data loader")

    df = pd.DataFrame(columns=['input_ids', 'label_id'])
    
    input_files = create_list_input_files(path)

    for file in input_files:
        df_temp = pd.read_csv(file, 
                              sep='\t', 
                              usecols=['input_ids', 'label_id'])
        df = df.append(df_temp)
        
    ds = ReviewDataset(
        input_ids_list=df.input_ids.to_numpy(),
        label_id_list=df.label_id.to_numpy(),
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    ), df
