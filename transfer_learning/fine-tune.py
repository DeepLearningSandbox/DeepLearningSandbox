import os
import sys
import glob
import argparse

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3
FILE_EXT = "jpg"
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172


def get_nb_files(dr, file_ext=FILE_EXT):
  """Get number of files ending in `file_ext` by searching directory dr recursively"""
  if not os.path.exists(dr):
    return 0
  fns = []
  for r, d, files in os.walk(dr):
    fns.extend([f for f in files if f.endswith(file_ext)])
  return len(fns)


def setup_to_transfer_learn(model):
  for layer in model.layers:
    layer.trainable = False


def add_new_last_layer(base_model, nb_classes):
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  # Freeze the bottom N layers and train the remaining top layers
  # we chose to train the top 2 inception blocks, i.e. we will freeze
  # the first 172 layers and unfreeze the rest:
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True

def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.nb_epoch)
  batch_size = int(args.batch_size)

  # setup model
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  setup_to_transfer_learn(base_model)
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

  train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  validation_generator = test_datagen.flow_from_directory(
    args.val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  h1 = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

  # fine-tuning
  setup_to_finetune(model)
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

  h2 = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

  model.save(args.output_model_file)


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir")
  a.add_argument("--val_dir")
  a.add_argument("--nb_epoch", default=NB_EPOCHS)
  a.add_argument("--batch_size", default=BAT_SIZE)
  a.add_argument("--output_model_file", default="inceptionv3-ft.model")

  args = a.parse_args()
  if args.train_dir is None or args.val_dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    print("directories do not exist")
    sys.exit(1)

  train(args)

