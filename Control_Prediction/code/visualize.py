import matplotlib.pyplot as plt

def visualize(img_path=None,gt_path=None,result_path=None,title=None):
    if result_path is None:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 8))
        fig.suptitle(title,size='x-large')
        for i in range(5):
            pic = plt.imread(img_path+"{}/{}_13.jpg".format(i+11,i+11))
            label = plt.imread(gt_path+"{}/{}_13.jpg".format(i+11,i+11))

            axes[0][i].axis('off')
            axes[0][i].imshow(pic)
            axes[0][i].set_title("Raw Image")

            axes[1][i].imshow(label,cmap='gray')
            axes[1][i].axis('off')
            axes[1][i].set_title("Ground Truth")
            
    else:
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 8))
        fig.suptitle(title,size='x-large')
        for i in range(5):
            pic = plt.imread(result_path+"{}_img.jpg".format(i))
            label = plt.imread(result_path+"{}_gt.jpg".format(i))
            result = plt.imread(result_path+"{}_pred.jpg".format(i))
                
            axes[0][i].axis('off')
            axes[0][i].imshow(pic)
            axes[0][i].set_title("Raw Image")

            axes[1][i].imshow(label,cmap='gray')
            axes[1][i].axis('off')
            axes[1][i].set_title("Ground Truth")

            axes[2][i].imshow(result,cmap='gray')
            axes[2][i].axis('off')
            axes[2][i].set_title("Prediction")