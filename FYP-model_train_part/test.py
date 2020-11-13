import tensorflow as tf
import PIL
from tensorflow.lite.python import lite
#from tensorflow.python.keras.layers.core import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.python.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import pathlib
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns
import numpy as np

train_path = "C:/Users/rosha/PycharmProjects/FYP/Cat-Man-Car/train"
validation_path = "C:/Users/rosha/PycharmProjects/FYP/Cat-Man-Car/validation"
test_path = "C:/Users/rosha/PycharmProjects/FYP/Cat-Man-Car/test"


train_batch = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_path, target_size=(224,224), batch_size=64)
valid_batch = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    validation_path, target_size=(224,224), batch_size=64)
test_batch = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    test_path, target_size=(224,224), batch_size=64, shuffle=False)


model = tf.keras.applications.mobilenet.MobileNet()  #call mobilenet API AND STORE INTO MOBILE variable
model.summary()

x = model.layers[-1].output
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

save_models = tf.keras.models.load_model('saved_model.h5', compile=False)
save_models.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
trained_model = model.fit_generator(train_batch, validation_data=test_batch, epochs=30)
loss,accuracy = save_models.evaluate(test_batch)
print(f'loss: {loss} and accuracy: {accuracy*100}')


# new_model = model.save("saved_model.h5")
test_labels = test_batch.classes  # this line shows classes and store into test_labels
print(test_labels)      # shows classes
print(test_batch.class_indices) # show class indices like 0 for car 1 for cat 2 for man
predictions = save_models.predict_generator(test_batch)
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
print(f'model input : {model.input}, model input_names : {model.input_names} ')
print(f'model output : {model.output}, model output_names : {model.output_names} ')


cm_plot_labels = ['car', 'cat', 'man']
plot_confusion_matrix(cm, cm_plot_labels, normalize=False)
import matplotlib.pyplot as plt
# plot training & validation accuracy values
plt.plot(save_models.history['accuracy'])
plt.plot(save_models.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='upper left')
plt.show()

# plot training & validation loss
plt.plot(save_models.history['loss'])
plt.plot(save_models.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='upper left')
plt.show()






