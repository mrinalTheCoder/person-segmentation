# Person Segmentation

The aim of this project is to separate the foreground (consisting of a person) from the background of an image. 
An image segmentation machine learning model is used for this task. 
Image segmentation means that the model classifies every single pixel in the image as foregrounnd/background.
In this problem the supervised learning label comprises of an image mask where each pixel corresponds to a target class.

### The Model
The U-net model architecture has been used for this task. It is an encoder-decoder model. 
The image is first encoded into a dense vector. From that, features are extracted for segmentation. 
An approximate trainging loss of 0.3 and an accuracy of 80% was achieved.

### Dataset used
The [Supervisely Person Dataset](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets) was used to train the model. 
This dataset can be downloaded only after signing into the site and should not be put into any commercial use.
Around a fifth of the total data (around 500 images) was used as the total size is too large. The data files are:
* `img.zip` [(training images)](https://drive.google.com/open?id=1NU-skxQaoE7CphcS_KfsuiqSXJG1sC1U)
* `masks.zip` [(training masks)](https://drive.google.com/open?id=1O1JeORPoGmVdRf8G_L9MME__rbp4xBph)
* `img_val.zip` [(validation images)](https://drive.google.com/open?id=1XZDIzBk_NqZk4ONK_rsL8mFFCyv_3BUX)
* `masks_val.zip` [(validation masks)](https://drive.google.com/open?id=149nbJpCK6Ezlg7HMxEIT-vJOwV-9iD0U)

# Output of the Model
The predictions of the model are not perfect. I am working on training with another model architecture, Tiramisu to get better predictions.
