# %%
import torch
import numpy as np
import pandas as pd

x = torch.normal((4000, 5))

x = x.numpy()
np.savetxt("firstarray.csv", x, delimiter=",")

# DF = pd.DataFrame(x)
 
# save the dataframe as a csv file
# DF.to_csv("data1.csv", index=False, header=False)
