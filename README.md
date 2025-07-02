# ![](https://github.com/HuiMengTingYu/SpaPred/blob/main/Logo/SpaPred.png)
SpaPred aims to automatically predict the spatiotemporal distribution patterns of hepatocellular carcinoma.
---

The heterogeneity of tumors, the diversity of tumor spatial structural characteristics, and the complexity of TIME make it extremely challenging to accurately analyze the composition and distribution of ST segment data in liver cancer patients. Therefore, the multi-algorithm fusion model (SpaPred) aimed at automatically generating the TIME subtype and tumor spatial structural information for each spot.

`Using SpaPred`

Load the Python modules required for the model and import the SpaPred model through the code.

* `import os` <br>
* `import pandas as pd` <br>
* `import numpy as np` <br>
* `import torch` <br>
* `from SpaPred import SpaPredModel` <br>

Use the built-in `load` function to load the data to be predicted, align the source data expression patterns through the `align_data` function, and finally predict the spatial structure and distribution patterns of immune microenvironment subtypes through the built-in `predict` function.
