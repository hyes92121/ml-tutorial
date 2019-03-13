# PyTorch tutorial

**TA tutorial,  Machine Learning (2019 Spring)**

## Contents
* Package Requirements
* NumPy Array Manipulation
* PyTorch
* Start building a model

## Package Requirement
**Note: This is a tutorial for `PyTorch==1.0.1` version**
* PyTorch == 1.0.1
* NumPy >= 1.14
* SciPy == 1.2.1

## NumPy Array Manipulation 
Some useful functions that you may use for managing your training data. We **must** carefully check our data dimensions are logically correct.

* `np.concatenate((arr_1, arr_2, ...), axis=0)`
  
   Note that the shape of array in the sequence should be the same except the dimension corresponds to the axis.
   
   ```python
       # concatenate two array
       a1 = np.array([[1, 2], [3, 4], [5, 6]])    # shape: (3, 2)
       a2 = np.array([[3, 4], [5, 6], [7, 8]])    # shape: (3, 2)

       # along the axis = 0
       a3 = np.concatenate((a1, a2), axis=0)      # shape: (6, 4)
   
       # along the axis = 1
       a4 = np.concatenate((a1, a2), axis=1)      # shape: (3, 4)
   ```
   
* `np.transpose(arr, axis)`
  
   Mostly we use it to align the dimension of our data.
   ```python
       # transpose 2D array
       a5 = np.array([[1, 2], [3, 4], [5, 6]])    # shape: (3, 2)
       np.transpose(a5)                           # shape: (2, 3)
   ```
   
   We can also permute multiple axis of the array.
   
   ```python
       a6 = np.array([[[1, 2], [3, 4], [5, 6]]])  # shape: (1, 3, 2)
       np.transpose((a6), axes=(2, 1, 0))         # shape: (2, 3, 1)
   ```
   
## PyTorch

### Tensor Manipulation

A `torch.tensor` is conceptually identical to a numpy array, but with GPU support and additional attributes to allow Pytorch operations. 

* Create a tensor

    ```python
        b1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    ```

* Some frequently-used functions you can use
    ```python
        b1.size()               # to check to size of the tensor
                                # torch.Size([1, 2, 3])
        b1.view((1, 3, 2))      # same as reshape in numpy (same underlying data, different interpretations)
                                # tensor([[[1, 2],
                                #          [3, 4],
                                #          [5, 6]]])
        b1.squeeze()            # removes all the dimensions of size 1 
                                # tensor([[1, 2, 3],
                                #         [4, 5, 6]])
        b1.unsqueeze()          # inserts a new dimension of size one in a specific position
                                # tensor([[[[1, 2, 3],
                                #           [4, 5, 6]]]])
    ```

* Other manipulation functions are similar to that of NumPy, we omitted it here for simplification. For more information, please check the PyTorch documentation: https://pytorch.org/docs/stable/tensors.html

###Tensor Attributes

- Some important attributes of `torch.tensor`

- ```python
        b1.grad                 # gradient of the tensor
        b1.grad_fn              # the gradient function the tensor
        b1.is_leaf              # check if tensor is a leaf node of the graph
        b1.requires_grad        # if set to True, starts tracking all operations performed
    ```

### Autograd

**torch.Auotgrad** is a package that provides functions implementing differentiation for scalar outputs.

For example:
* Create a tensor and set `requires_grad=True` to track the computation with it.

    ```python
        x1 = torch.tensor([[1., 2.],
                           [3., 4.]], requires_grad=True)
      # x1.grad             None 
      # x1.grad_fn          None
      # x1.is_leaf          True
      # x1.requires_grad    True
        
        x2 = torch.tensor([[1., 2.],
                           [3., 4.]], requires_grad=True)
      # x2.grad             None 
      # x2.grad_fn          None
      # x2.is_leaf          True
      # x2.requires_grad    True
    ```

    It also enables the tensor to do gradient computations later on.

    Note: Only floating dtype can require gradients.

* Do some simple operation

    ```python
        z = (0.5 * x1 + x2).sum()
      # x2.grad             None 
      # x2.grad_fn          <SumBackward0>
      # x2.is_leaf          False
      # x2.requires_grad    True
    ```

    Note: If we view `x1` as 

    ![equation](https://latex.codecogs.com/svg.latex?%5Clarge%20X_1%3D%20%5Cleft%5B%20%7B%5Cbegin%7Barray%7D%7Bcc%7D%20x_1%20%26%20x_2%20%5C%5C%20x_3%20%26%20x_4%20%5C%5C%20%5Cend%7Barray%7D%20%7D%20%5Cright%5D)

    ​           and view `x2` as 

    ![equation](https://latex.codecogs.com/svg.latex?%5Clarge%20X_2%3D%20%5Cleft%5B%20%7B%5Cbegin%7Barray%7D%7Bcc%7D%20x_5%20%26%20x_6%20%5C%5C%20x_7%20%26%20x_8%20%5C%5C%20%5Cend%7Barray%7D%20%7D%20%5Cright%5D)

    ​           Then `z` is equvilant to ![equation](https://latex.codecogs.com/svg.latex?%5Clarge%20z%3D%5Cfrac%7B1%7D%7B2%7D%28x_1&plus;x_2&plus;x_3&plus;x_4%29&plus;%28x_5&plus;x_6&plus;x_7&plus;x_8%29)

* Call `backward()` function to compute gradients automatically

    ```python
        z.backward()	# this is identical to calling z.backward(torch.tensor(1.))
    ```

    `z.backward()` is actually just the derivative of z with respect to inputs (tensors whose `is_leaf` and `requires_grad` both equals `True`)

    For example, if we want to know the derivative of `z` with respect to `x_1`, it is:

    ![equation](https://latex.codecogs.com/svg.latex?%5Clarge%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x_1%7D%3D0.5)

* Check the gradients using `.grad`

    ```python
        x1.grad
        x2.grad
    ```

    Output will be something like this

    ```python
        tensor([[[0.5000, 0.5000],        # x1.grad
                 [0.5000, 0.5000]]])
        tensor([[[1., 1.],                # x2.grad
                 [1., 1.]]])
    ```

More in-depth explanation of Autograd can be found in this awesome youtube video: [Link](https://youtu.be/MswxJw-8PvE)

## Start building a model

### Dataset class

Pytorch provides a convenient way for interacting with datasets by `torch.utils.data.Dataset`, an abstract class representing a dataset. When datasets are large, the RAM on our machine may not be large enough to fit all the data at once. Instead, we load only a portion of the data when needed, and move it back to disk when finished using.

A simple dataset is created as follows:	

```python
import csv 
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, label_path):
        """
        let's assume the csv is as follows:
        ================================
        image_path                 label
        imgs/001.png               1     
        imgs/002.png               0     
        imgs/003.png               2     
        imgs/004.png               1     
                      .
                      .
                      .
        ================================
       	And we define a function parse_csv() that parses the csv into a list of tuples 
       	[('imgs/001.png', 1), ('imgs/002.png', 0)...]
        """		
        self.label = parse_csv()
       
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path, label = self.label[idx]
       	
        # imread: a function that reads an image from path
        
        img = imread(img_path)
        
        # some operations/transformations
        
        return torch.tensor(img), torch.tensor(label)
        
```

Note that `MyDataset` inherits `Dataset`. If we look at the [source code](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset), we can see that the default behavior of `__len__` and `__getitem__` is to raise a `NotImplementedError`, meaning that we should override them every time we create a custom dataset. 

###Dataloader

We can iterate through the dataset with a `for` loop, but we cannot shuffle, batch or load the data in parallel. `torch.utils.data.Dataloader` is an iterator which provides all those features. We can specify the batch size, whether to shuffle the data, and number of workers to load the data. 

```python
from torch.utils.data import DataLoader

dataset = MydDataset('/imgs')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch_id, batch in enumerate(dataloader):
    imgs, labels = batch
    
    """
    do something for each batch
    ex: 
        output = model(imgs) 
        loss = cross_entropy(output, labels)
    """

```

### Model

Pytorch provides an `nn.Module` for easy definition of a model. A simple CNN model is defined as such:

```python
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__() # call parent __init__ function
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )
        self.output = nn.Softmax(dim=1)
       
    def forward(self, x):
        # You can modify your model connection whatever you like
        out = self.fc(x.view(-1, 28*28))
        out = self.output(out)
        return out        
```

We let our model inherit from the `nn.Module` class. But why do we need to call `super` in the `__init__` function whereas in the `Dataset` case we don't ? If we look at the [source code](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module) of `nn.Module` we can see that there are certain attributes needed in order for the model to work. In the case of `Dataset`, there is no `__init__` function, so no `super` is needed.

In addition, `forward` is also by default not implemented, so we need to override it with our own forward propagation function. 

### Example

A full example of a MNIST classifier: [Link](https://github.com/fanoping/ml-pytorch-tutorial/blob/master/mnist_pytorch.ipynb)

