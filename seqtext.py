import torch
import torch.nn as nn
class TextModule(nn.Module):
    def __init__(self,p1,p2,p3):
        super(TextModule,self).__init__()
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.W1=nn.Linear(p1,p2,bias=False)
        self.B1=nn.Parameter(torch.ones([p2]))
        self.W2=nn.Linear(p2,p1,bias=False)
        self.B2=nn.Parameter(torch.ones([p1]))
        self.W=nn.Linear(p1,p3,bias=False)
        self.B=nn.Parameter(torch.ones([p3]));
    def forward(self,X):
        g1=self.W1(X)+self.B1
        g2=self.W2(g1)+self.B2
        g3=self.W(g2)+self.B
        return g3
class TextModule2(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        if a>b: raise ValueError
        self.a=a;
        self.b=b;
        self.W1=nn.Linear(a,a);
    def forward(self,X):
        g=self.W1(X);
        k=torch.ones([self.b-self.a]);
        return torch.cat([g,k]);

if __name__=="__main__":
    # model=nn.Sequential(
    #     TextModule(3,6,4),
    #     TextModule(4,5,3)
    # )
    model=nn.Sequential()
    model.add_module("1",TextModule(3,6,4))
    model.add_module("2",TextModule(4,5,3))
    tx=torch.Tensor([1,2,3])
    a=TextModule(3,6,4)
    print(a(tx))
    print(model(tx));
    model2=TextModule2(3,5);
    print(model2(tx));
    k=torch.Tensor([[2,3,4],[5,6,7]]);
    g=torch.Tensor([[1,1,1,1],[1,1,1,1]]);
    print(k.shape);print(g.shape);
    print(torch.cat([k,g],dim=1));