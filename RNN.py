import torch

class linear:
    def __init__(self, l1_size, l2_size):
            self.l1 = torch.zeroes(l1_size, dtype = torch.float)     
            self.l2 = torch.zeros(l2_size, dtype = torch.float)
            self.W = torch.normal(0, torch.sqrt(torch.tensor(2.0/l1_size)), size=(l2_size, l1_size))
            self.b = torch.zeros(l2_size, dtype=torch.float) + 0.01
    
class RNN:
    def __init__(self, input_size, output_size):
        self.fc1 = linear(input_size, 32)
        self.fc2 = linear(self.fc1.l2, 32)
        self.fc3 = linear(self.fc2.l2, output_size)

    def forward_pass(self, x):
         self.fc1.l1.copy_(x)

         self.fc1.l2 = torch.matmul(self.fc1.W, self.fc1.l1) + self.fc1.b
         self.fc1.l2 = torch.max(torch.zeros_like(self.fc1.l2), self.fc1.l2)

         self.fc2.l2 = torch.matmul(self.fc2.W, self.fc2.l1) + self.fc2.b
         self.fc2.l2 = torch.max(torch.zeros_like(self.fc2.l2), self.fc2.l2)

         self.fc3.l2 = torch.matmul(self.fc3.W, self.fc3.l1) + self.fc3.b
         self.fc3.l2 = torch.max(torch.zeros_like(self.fc3.l2), self.fc3.l2)


    def relu_derivative(x):
        return (x > 0).float()
    
    def backward_pass(self, y):
        dLdW = relu_derivative(2*(self.fc3.l2 - y))

    def weight_adj():
        
    def bias_adj():
        



         
         




              
              
         

