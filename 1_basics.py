import torch

# instructions from Youtube video https://www.youtube.com/watch?v=c36lUUr864M&list=WL&index=4
# from Youtube channel https://www.youtube.com/@patloeber

a = torch.tensor([2.1, 4])
x = torch.rand(2, 2)
y = torch.rand(2, 2)
z = torch.add(x, y)
print(z[1, :])
# resizing
x = torch.rand(4, 4)
x = x.view(2, 8)
print(x)

