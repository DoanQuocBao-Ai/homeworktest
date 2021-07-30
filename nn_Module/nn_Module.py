import torch
import math
class Polymonial(torch.nn.Module):
    def __init__(selt):
        super().__init__()
        selt.a=torch.nn.Parameter(torch.randn(()))
        selt.b=torch.nn.Parameter(torch.randn(()))
        selt.c=torch.nn.Parameter(torch.randn(()))
        selt.d=torch.nn.Parameter(torch.randn(()))
    def forward(selt,x):
        return selt.a*x**3+selt.b*x**2+selt.c*x+selt.d
    def string(selt):
        return selt.a.item(),"x^3",selt.b.item(),"x^2",selt.c.item(),"x",selt.d.item()

x=torch.linspace(-math.pi,math.pi,2000)
y=torch.sin(x)

modul=Polymonial()
cri=torch.nn.MSELoss(reduction="sum")
optim=torch.optim.SGD(modul.parameters(),lr=1e-6)
for i in range(1000):
    y_pred=modul(x)
    loss=cri(y_pred,y)
    print(i,"loss",loss.item())
    cri.zero_grad()
    loss.backward()
    optim.step()
print("Module: ",modul.string())