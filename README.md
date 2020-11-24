# Machine Learning Nanodegree
# Deep Learning
## Project: Deep Learning Image Classifier

Project code for Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. In this project, I developed code for an image classifier built with TensorFlow, then converted it into a command line application.

In order to complete this project, I used the GPU enabled workspaces within the Udacity classroom.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [tensorflow](https://www.tensorflow.org/api_docs)
- [tensorflow_hub](https://www.tensorflow.org/hub)
- [tensorflow_datasets](https://www.tensorflow.org/datasets)
- [json](https://docs.python.org/3/library/json.html)
- [time](https://docs.python.org/3/library/time.html)
- [os](https://docs.python.org/3/library/os.html)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

I recommend installion [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

Template code is provided in the `Project_Image_Classifier_Project.ipynb` notebook file.

### Run

In a terminal or command window, navigate to the top-level project directory `Deep-Learning-Image-Classifier/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Project_Image_Classifier_Project.ipynb
```  
or
```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

This will open the iPython Notebook software in your browser.

To use the command line implementation in a terminal or command window, navigate to the top-level project directory `Deep-Learning-Image-Classifier/` (that contains this README) and run one of the following commands:

```bash
python predict.py path/to/image path/to/model 
```

**Examble:**

```bash
python predict.py test_images/cautleya_spicata.jpg best_model.h5
```

There are also some optional parameters:-
- --top_k returns: the top k classes with their probabilties.
- --category_names: a json file that maps each class number with a class name

**Examble:**

```bash
python predict.py test_images/cautleya_spicata.jpg best_model.h5 --top_k 6 --category_names label_map.json
```

**Data**

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github. If you would like the data for this project, you will want to download it from the workspace in the classroom.
```bash
%pip --no-cache-dir install tensorflow-datasets --user
%pip --no-cache-dir install tfds-nightly --user
```
Though actually completing the project is likely not possible on your local machine unless you have a GPU. I trained the deep learning classifier using 102 different types of flowers, where there ~20 images per flower to train on. Then I used the trained classifier to see if I can predict the type for new images of the flowers.

### Certification
<p align="middle"><a href="https://github.com/Omar-Al-Khathlan/Deep-Learning-Image-Classifier/blob/main/Certificate/Udacity%20Certificate.pdf"><img src="https://github.com/Omar-Al-Khathlan/Deep-Learning-Image-Classifier/blob/main/Certificate/Udacity%20Certificate.png"/></a></p>
