import numpy as np

def load_optical_flow(path):
    flow = np.load(path)
    return flow

def save_optical_flow(flow, path):
   try:
      np.save(path, flow)
      return True
   except:
      return False
  
# Just for testing  
if __name__ == '__main__':
   flow = np.random.rand(20, 20, 2)
   print(load_optical_flow("./output_flow/flow_fire_1.npy"))
   