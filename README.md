# Football Training Data Creator

Finally, a solution to picking up the remote control and muting live TV


## End to End Training

The following assumes videos somewhere on the filesystem with the
following directory structure:

```
- some_dir
- - ad
- - game
- - messedup
-- converted_images
```

Videos used for training should be in TODO format and separated into their
appropriate `ad/` and `game/` directories. Any videos that have both ad and
game footage should be moved to messedup until they can be edited and
disambiguated.

First, convert the videos to images (jpgs) for training (cd into the video
subdirectory of this project):

```bash
# NOTE: Be sure to set the environment variables in the docker-compose
# file correctly.
docker-compose run --rm video_converter
```

This will resize and grayscale the frames of the video and store them in
`converted_images`


Training happens as a series of steps in the `model_training` in the 
`docker-compose`. Comment/Uncomment the `command` lines in the `model_training`
service one at a time until a model is produced. **These steps depend on the queue
running (see the section about starting the queue).**

Each file's purpose is documented in the module docstring.

At the completion of running each module, all the training data should be serialized
in the `models_and_training_data` directory. A model for prediction will also
be stored in this directory.

## Prediction

Prediction happens by writing a 320x240, grayscaled, base64 encoded image to the
Kafka queue. A consumer of the queue reads the image and classifies it.

To start the consumer, run this:

```bash
docker-compose run --rm image_stream_classifier
```

Double check that the volumes are mapped correctly.


## Queue

This project depends on Kafka. To start up the queue (prior to doing any training
or prediction!), cd into the `queue/` directory and run:

```bash
docker-compose up
```

All required topics are defined in the docker-compose file.
