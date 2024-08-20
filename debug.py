from ultralytics import YOLO
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
train
"""
# classify
# weight_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\classify\weights\best.pt"
# cfg_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\classify\yolov8-cls.yaml"
# data_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\data\classify"
# model = YOLO(cfg_path).load(weight_path) 
# results = model.train(data=data_path, 
#             imgsz=640, 
#             batch=16,    
#             epochs=100,
#             name="yolov8-cls") 

# detect
# weight_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\detect\weights\best.pt"
# raw_weight_path = r"E:\Project\AutoAI\AAIstandardExamples(encryptmodel)\model\detection\lidian\rawweights.pt"
# cfg_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\detect\yolov8-seg.yaml"
# data_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\data\detect\coco128-seg.yaml"
# # model = YOLO(cfg_path).load(raw_weight_path)
# model = YOLO(weight_path)
# results = model.train(data=data_path, 
#             imgsz=640, 
#             batch=4,    
#             epochs=100,
#             workers=0,
#             amp=False,
#             name="yolov8-det") 


# segment
# weight_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\segment\weights\best.pt"
# cfg_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\segment\yolov8-seg.yaml"
# data_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\data\segment\coco128-seg.yaml"
# model = YOLO(cfg_path).load(weight_path) 
# results = model.train(data=data_path, 
#             imgsz=640, 
#             batch=16,    
#             epochs=100,
#             workers=0,
#             amp=False,            
#             name="yolov8-seg") 




"""
val
"""
# classify
# model = YOLO(r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\classify\weights\best.pt")
# metrics = model.val(data=r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\data\classify\ImageNet.yaml", 
#                     imgsz=640, 
#                     batch=16,
#                     conf=0.25) 
# print(metrics)

# detect
# model = YOLO(r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\detect\weights\best.pt")
# metrics = model.val(data=r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\data\detect\coco128-seg.yaml", 
#                     imgsz=640, 
#                     batch=16,
#                     conf=0.25) 
# print(metrics)

# model = YOLO(model='yolov8n.yaml').load(r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\detect\weights\best.pt")
# model.model.eval()
# img = torch.randn(2, 3, 640, 640)
# results = model(img)
# print(results)

# segment
# model = YOLO(r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\segment\weights\best.pt")
# metrics = model.val(data=r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\data\segment\coco128-seg.yaml", 
#                     imgsz=640, 
#                     batch=16,
#                     conf=0.25) 
# print(metrics)



"""
export
"""
# classify
weight_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\classify\weights\best.pt"
model = YOLO(weight_path)
success = model.export(format="TorchScript", jit_train_mode=True, device='cuda') 

# detect
# weight_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\detect\weights\best.pt"
# cfg_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\detect\yolov8-seg.yaml"
# model = YOLO(weight_path)
# success = model.export(format="TorchScript", jit_train_mode=True, device='cuda') 

# segment
# weight_path = r"E:\LGJ\program\yolov8\runs\segment\yolov8-seg16\weights\best.pt"
# model = YOLO(weight_path)
# success = model.export(format="TorchScript", jit_train_mode=True, device='cuda') 

# 测试模型
# train_model_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\classify\weights\best_train.torchscript"
# test_model_path = r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\classify\weights\best_test.torchscript"
# train_model = torch.jit.load(train_model_path)
# test_model = torch.jit.load(test_model_path)
# results1 = train_model(img)
# results2 = test_model(img)
# model = YOLO(r"E:\Project\AutoAI\AAIstandardExamples(rawmodel)\lidian\yolov8\raw_pt\detect\weights\best.pt")
# img = torch.rand(1, 3, 640, 640)
# model.model.train()
# results3 = model.model(img)
# model.model.eval()
# results4 = model.model(img)
# pass