import torch  # 命令行是逐行立即执行的


content = torch.load('experiments/bridge_pretrained_2023-05-26_10-20-17_ZNU/checkpoints/model_best.pth')
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
print(content['model'])
