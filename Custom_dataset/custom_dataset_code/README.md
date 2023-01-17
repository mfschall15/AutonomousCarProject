# Dataset Creation

**collect-images-lane-follow.py:**
Collects images from simulator and saves into specific file structure. Also writes control input to csv

**Convert_binary_tvt.py:** Converts tvt lane dataset into binary images using crude opencv operations.

**Convert_binary.py:** Converts simulator lane dataset into binary images using opencv operations.

**edges_detection.py:** Utility functions used by the binary image scripts to detect lane lines

**stats_calc.py:** Script used to calculatoe precision, Accuracy, Recall, and F1 of binary lane images over the ground truth.

