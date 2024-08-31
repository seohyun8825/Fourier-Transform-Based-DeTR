import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageDraw 

class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None):
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.coco['images'])

    def __getitem__(self, idx):
        img_info = self.coco['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        anns = [ann for ann in self.coco['annotations'] if ann['image_id'] == img_info['id']]
        boxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        # 원본 이미지 크기
        original_size = img.size

        if self.transforms is not None:
            img = self.transforms(img)


        transformed_size = img.shape[1:]  # (channels, height, width)

        boxes = self.adjust_boxes(boxes, original_size, transformed_size)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = img_info['id']
        target['class_labels'] = torch.tensor(labels, dtype=torch.int64)

        return img, target

    def adjust_boxes(self, boxes, original_size, transformed_size):
        original_width, original_height = original_size
        transformed_height, transformed_width = transformed_size

        new_boxes = []
        for box in boxes:
            x_min, y_min, width, height = box
            x_min = (x_min / original_width) * transformed_width
            y_min = (y_min / original_height) * transformed_height
            width = (width / original_width) * transformed_width
            height = (height / original_height) * transformed_height
            new_boxes.append([x_min, y_min, width, height])

        return new_boxes

def get_dataloader(annotation_file, img_dir, batch_size=4, shuffle=True):
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])

    dataset = CocoDataset(annotation_file, img_dir, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))
    return dataloader

def visualize_and_save_sample_images(dataloader, save_dir="visualized_samples", num_samples=5):
    os.makedirs(save_dir, exist_ok=True)

    for i, (images, targets) in enumerate(dataloader):
        if i >= num_samples:
            break

        for j, (image, target) in enumerate(zip(images, targets)):

            img = T.ToPILImage()(image)
            draw = ImageDraw.Draw(img)

            for box in target['boxes']:
                # [x_min, y_min, width, height]
                x_min, y_min, width, height = box
                x_max = x_min + width
                y_max = y_min + height

                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

            img.save(os.path.join(save_dir, f"sample_{i * len(images) + j}.png"))
            print(f"Saved: {os.path.join(save_dir, f'sample_{i * len(images) + j}.png')}")


#dataloader = get_dataloader('data/train_annotations.json', 'data/train_images', batch_size=4, shuffle=True)
#visualize_and_save_sample_images(dataloader)
