import torch
import torchvision
from model import MyAwesomeModel

#model = torchvision.models.resnet18(pretrained=True)

model = MyAwesomeModel()
model.load_state_dict(torch.load('models/checkpoint.pth'))
script_model = torch.jit.script(model)
script_model.save('models/scripted.pt')

testloader = torch.load('data/processed/test_loader.pth')

image, labels = next(iter(testloader))
for idx,img in enumerate(image):
    img = img.view(1,28,28) 
    
    log_ps = model(img)
    log_ps_sc = script_model(img)
    _, top_class = torch.exp(log_ps).topk(1, dim=1)
    _, top_class_sc = torch.exp(log_ps_sc).topk(1, dim=1)
print(img)
print(log_ps,top_class)
print(log_ps_sc,top_class_sc)
assert torch.allclose(log_ps,log_ps_sc)