from skimage.color import rgb2gray, rgba2rgb
from util.util import tensor2im 
import numpy as np
import torch
import torch.nn.functional as f

class LBP_Extractor():

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    self.output_nc = opt.output_nc
    
  def get_pixel(img, center, x, y): 

      new_value = 0

      try: 
          # If local neighbourhood pixel  
          # value is greater than or equal 
          # to center pixel values then  
          # set it to 1 
          if img[x][y] >= center: 
              new_value = 1

      except: 
          # Exception is required when  
          # neighbourhood value of a center 
          # pixel value is null i.e. values 
          # present at boundaries. 
          pass

      return new_value 

  # Function for calculating LBP 
  def lbp_calculated_pixel(img, x, y): 
      # print((img[x][y]).shape)
      center = img[x][y] 

      val_ar = [] 

      # top_left 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x-1, y-1)) 

      # top 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x-1, y)) 

      # top_right 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x-1, y + 1)) 

      # right 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x, y + 1)) 

      # bottom_right 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x + 1, y + 1)) 

      # bottom 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x + 1, y)) 

      # bottom_left 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x + 1, y-1)) 

      # left 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x, y-1)) 

      # Now, we need to convert binary 
      # values to decimal 
      power_val = [1, 2, 4, 8, 16, 32, 64, 128] 

      val = 0

      for i in range(len(val_ar)): 
          val += val_ar[i] * power_val[i] 

      return val 


  def toLBP(x, is_tensor = True, output_nc = 1, back2tensor = True): 
    # if is_tensor == True:
    #   _,l, h, w = x.shape
    #   x = torch.reshape(x, (l,h, w))

    # [_, _, Rows,Cols]=x.shape
    # x=x.reshape(3,Rows,Cols)

    paddings = (0,0,1,1,0,0)
    # print(x.shape)
    x=f.pad(x, paddings,"constant", 0)        
    b=x.shape 
    M=b[1]
    N=b[2]      
    y=x
    #select the pixels of masks in the form of matrices
    y00=y[:,0:M-2, 0:N-2]
    y01=y[:,0:M-2, 1:N-1]
    y02=y[:,0:M-2, 2:N  ]
    #     
    y10=y[:,1:M-1, 0:N-2]
    y11=y[:,1:M-1, 1:N-1]
    y12=y[:,1:M-1, 2:N  ]
    #
    y20=y[:,2:M, 0:N-2]
    y21=y[:,2:M, 1:N-1]
    y22=y[:,2:M, 2:N ]  

    # Comparisons 
    # 1 -------------------------------        
    g=torch.greater_equal(y01,y11)
    z=torch.multiply(g.type(torch.FloatTensor), 
                  torch.tensor(1) )      
    # 2 ---------------------------------
    g=torch.greater_equal(y02,y11)
    tmp=torch.multiply(g.type(torch.FloatTensor), 
                    torch.tensor(2) )
    z =torch.add(z,tmp)              
    # 3 ---------------------------------
    g=torch.greater_equal(y12,y11)
    tmp=torch.multiply(g.type(torch.FloatTensor), 
                    torch.tensor(4) )
    z =torch.add(z,tmp)
    # 4 ---------------------------------
    g=torch.greater_equal(y22,y11)
    tmp=torch.multiply(g.type(torch.FloatTensor), 
                    torch.tensor(8) )
    z =torch.add(z,tmp)  
    # 5 ---------------------------------
    g=torch.greater_equal(y21,y11)
    tmp=torch.multiply(g.type(torch.FloatTensor), 
                    torch.tensor(16) )
    z =torch.add(z,tmp)  
    # 6 ---------------------------------
    g=torch.greater_equal(y20,y11)
    tmp=torch.multiply(g.type(torch.FloatTensor), 
                    torch.tensor(32) )
    z =torch.add(z,tmp)  
    # 7 ---------------------------------
    g=torch.greater_equal(y10,y11)
    tmp=torch.multiply(g.type(torch.FloatTensor), 
                    torch.tensor(64) )
    z =torch.add(z,tmp)  
    # 8 ---------------------------------
    g=torch.greater_equal(y00,y11)
    tmp=torch.multiply(g.type(torch.FloatTensor), 
                    torch.tensor(128) )
    z =torch.add(z,tmp)  
    #---------------------------------   
    z = z.type(torch.FloatTensor) 
    # print(z.shape)
    if(back2tensor):
      return z.to('cuda')
    # print(z.shape)
    z = tensor2im(z)
    # print(z.shape)
    # if is_tensor == True:
    #   _,height, width, _ = z.shape
    # else:
    _, height, width = z.shape
    z = z.reshape(height,width)
    return z

  def toLBP2(tensor_bgr, is_tensor = True, output_nc = 1, back2tensor = True):
    
    # print('tensor_bgr tensor: ', tensor_bgr.shape)
    img_bgr = tensor2im(tensor_bgr)
    if is_tensor == True:
      #tensor_bgr = torch.unsqueeze(tensor_bgr, 0)
      
      # print('img_bgr tensor: ', img_bgr.shape)
      _,height, width, _ = img_bgr.shape
      
    else:
      #img_bgr = tensor_bgr
      # print('img_bgr non-tensor: ', img_bgr.shape)
      _, height, width = img_bgr.shape
      
      
    

      #    if output_nc == 1:
      #      
      #    else:
      
      
    #print('img_bgr:', img_bgr.shape)
    # grayscale = rgb2gray(rgba2rgb(img_bgr))
    grayscale = rgb2gray(img_bgr)
    # print('grayscale:', grayscale.shape)
    grayscale = grayscale.reshape(height, width)
    # print('grayscale after reshape:', grayscale.shape)
    # print(grayscale.size)
    # print(height)
    # print(width)
    img_lbp = np.zeros((height, width), 
                      np.uint8)
    # print('img_lbp:', img_lbp.shape)                  
    for i in range(0, height): 
      for j in range(0, width): 
        img_lbp[i, j] = LBP_Extractor.lbp_calculated_pixel(grayscale, i, j)

    if(back2tensor):
      tensor = torch.tensor(img_lbp)
      tensor_cuda = tensor.to('cuda')
      tensor_full = torch.reshape(tensor_cuda, (1, 1, height, width))
      return tensor_full
    return img_lbp