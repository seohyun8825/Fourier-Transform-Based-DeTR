import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from transformers import DetrForObjectDetection, AutoImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scripts.data_processing import get_dataloader
from models.backbone import MY_BACKBONE
from models.neck import My_Neck

import json

batch_size = 16
learning_rate = 1e-4
epochs = 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


train_loader = get_dataloader('data/train_annotations.json', 'data/train_images', batch_size=batch_size, shuffle=True)
val_loader = get_dataloader('data/val_annotations.json', 'data/val_images', batch_size=batch_size, shuffle=False)


backbone = MY_BACKBONE(base_channels=64, base_depth=2, deep_mul=1.0)
neck = My_Neck()
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.backbone = backbone
model.neck = neck
model.to(device)


optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


coco_gt = COCO('data/val_annotations.json')


def evaluate(model, val_loader, coco_gt):
    model.eval()
    coco_results = []
    image_ids = []

    print("Starting evaluation...")
    
    for images, targets in val_loader:
        images = torch.stack(images).to(device)
        outputs = model(pixel_values=images)

        for i, output in enumerate(outputs.logits):
            scores = output.softmax(-1)[..., :-1].max(-1)[0]
            labels = output.softmax(-1)[..., :-1].max(-1)[1]
            boxes = outputs.pred_boxes[i].detach().cpu().numpy()

            image_id = targets[i]["image_id"].cpu().item()
            image_ids.append(image_id)

            for box, score, label in zip(boxes, scores, labels):
                box = box.tolist()
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": float(score)
                })

    with open('outputs/detection_results.json', 'w') as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes('outputs/detection_results.json')
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    print("Evaluation complete.")
    return coco_eval.stats[0]  # mAP

# 학습 루프
for epoch in range(epochs):
    print(f"Epoch [{epoch + 1}/{epochs}] starting...")

    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(train_loader):
        images = torch.stack(images).to(device)
        
        # Debugging print
        print(f"Batch {i + 1}: Loaded {len(images)} images.")

        # 텐서 변환과 디버깅 정보 출력
        for t in targets:
            t['boxes'] = torch.tensor(t['boxes'], dtype=torch.float32).to(device) if not isinstance(t['boxes'], torch.Tensor) else t['boxes'].clone().detach().to(device)
            t['labels'] = torch.tensor(t['labels'], dtype=torch.int64).to(device) if not isinstance(t['labels'], torch.Tensor) else t['labels'].clone().detach().to(device)
            t['class_labels'] = torch.tensor(t['class_labels'], dtype=torch.int64).to(device) if not isinstance(t['class_labels'], torch.Tensor) else t['class_labels'].clone().detach().to(device)

        outputs = model(pixel_values=images, labels=targets)
        loss = outputs.loss
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {i + 1}: Loss = {loss.item()}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] completed with average loss: {avg_loss}")

    map_score = evaluate(model, val_loader, coco_gt)
    print(f"Epoch [{epoch + 1}/{epochs}], mAP: {map_score:.4f}")

# 모델 저장
model.save_pretrained("outputs/detr_lftdet_model")
print("Model saved successfully.")