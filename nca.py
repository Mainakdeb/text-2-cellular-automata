import torch

sobel_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0],],dtype=torch.float32)
laplacian_filter = torch.tensor([[1.0,2.0,1.0], [2.0,-12,2.0], [1.0,2.0,1.0]])

def perception(x):
  filters = torch.stack([identity_filter, sobel_filter, sobel_filter.T, laplacian_filter])
  return perchannel_conv(x, filters)

def perchannel_conv(x, filters):
  '''filters: [filter_n, h, w]'''
  b, ch, h, w = x.shape
  y = x.reshape(b*ch, 1, h, w)
  y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
  y = torch.nn.functional.conv2d(y, filters[:,None])
  return y.reshape(b, -1, h, w)
  
class ca_model(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=96):
    super().__init__()
    self.chn = chn
    self.w1 = torch.nn.Conv2d(chn*4, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()

  def forward(self, x, update_rate=0.5):
    y = perception(x)
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    return x+y*udpate_mask

  def seed(self, n, sz=128):
    seed=torch.randn(n, self.chn, sz, sz)
    # #normal
    seed -= seed.min(1, keepdim=True)[0]
    seed /= seed.max(1, keepdim=True)[0]
    # seed=torch.zeros(n, self.chn, sz, sz)
    # print("min ", torch.min(seed), "max ", torch.max(seed))
    return seed