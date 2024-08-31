import torch
import json
import os
from PIL import Image
from transformers import DetrForObjectDetection, AutoImageProcessor

# 모델 불러오기
model = DetrForObjectDetection.from_pretrained("outputs/detr_lftdet_model")
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# 테스트 이미지 경로 설정
test_img_dir = 'data/test_images'
test_image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]

# 예측 결과 저장할 리스트
submission_results = []

for image_file in test_image_files:
    image_path = os.path.join(test_img_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    image_id = int(image_file.split('.')[0])  # Assuming image ID is the filename without extension
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        result = {
            "image_id": image_id,
            "category_id": label.item(),
            "bbox": box,
            "score": round(score.item(), 3),
            "segmentation": []
        }
        submission_results.append(result)

# JSON 파일로 저장
with open('outputs/submission.json', 'w') as f:
    json.dump(submission_results, f)
