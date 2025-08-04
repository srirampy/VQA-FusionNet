import torch
from torch.utils.data import Dataset
from PIL import Image

class VQADataset(Dataset):
    def __init__(self, data, classes_to_idx, transform, vit_model, image_folder):
        self.data = data
        self.classes_to_idx = classes_to_idx
        self.transform = transform
        self.vit_model = vit_model.eval()
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(f"{self.image_folder}/{row['image_id']}.jpg").convert("RGB")
        text = row['question']
        label = self.classes_to_idx[row['answer']]

        if self.transform:
            image = self.transform(image)

        with torch.no_grad():
            img_feature = self.vit_model(image.unsqueeze(0)).squeeze(0)

        return img_feature, row['text_embedding'], torch.tensor(label)
