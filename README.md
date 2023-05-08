# Shear Sense Data Modelling


## Software Overview


### Project summary
The shear-sense project explores the use of touch sensor arrays that can detect both pressure and shear for gesture classification. The goal is to evaluate the usefulness of shear data through gesture classification, with the ultimate aim of collecting richer touch data for future human-computer interaction and human-robot interaction studies. The project involves using the touch sensor arrays to collect both pressure and shear data for a series of predetermined gestures performed by recruited participants, and then attempting to classify these gestures using machine learning techniques.


### purpose of the software in context of the project
The preprocessing pipeline takes the raw data collected by the touch sensor arrays and transforms it into video and pickled format in which we can analyze. These videos served as the training and testing set for the CNN model.


## Data processing pipeline


### Set-up
- Install required packages through `pip install -r requirements.txt`
- The 2 data processing pipelines can be run by `python data_processor.py` and `python data_processor.py`, respectively. The difference is explained below.
- The video output directory, the participant(s) data to analyze, and the experimental condition to analyze can be specified by the call to the function main.
   ```
   if __name__ == "__main__":
       main(<video data directory>, <list of participant numbers>, <experimental condition>)
   ```
   For example, generating video data in a directory called `video_data` with data from the first 10 participants on the *flat* condition can be done by changing the above segment to
   ```
   if __name__ == "__main__":
       main("video_data",range(1,10), "FLAT")
   ```


### Input format
The current implementation of the code requires the following input format:
- The participant data files are stored in a directory named `data` *outside* the root directory
   ```
   modelling/ <-- project root
   ├── data_processor_2.py
   ├── data_processor.py
   ├── ...
   data/
   ├── P1_CURVED.csv/
   ├── ...
   ```
- The individual CSV files must have the following naming scheme `P{participant number}_*{type}*.csv`, where `participant number` is a number, `type` is one of "CURVED" or "FLAT", and `*` is a wildcard that can be any character.


   If you wish to change ths above scheme, you may alter the following code segment to specify a different directory/filename globbing. `...sorted(glob.glob(os.path.join(cwd, f"../data/P{i}_*{type}*csv")))`.


- Each CSV file in the data directory must have 183 columns with the following format:
   - The first column of the data represents the timestamp for each row of data, indicating the exact time at which the touch data was recorded.
   - Columns 2-181 of the data represent the raw data for each individual taxel on the touch sensor array.
   - Column 182 of the data contains a touch/no-touch classification represented by "1"/"0", which indicates whether or not each taxel is currently being touched.
   - Column 183 of the data contains a label, this can be used for indicating the current gesture being performed at that time.


   The first row of the data represents the baseline state of the touch sensor array, and only columns 2-181 are populated with data. All subsequent rows show raw serial data collected by the sensor at the specified timestamp, and all 183 columns should be populated.


### Output format
Video (version 1)
- This version of video is generated by running `data_processor.py`. The schema in `model.py` is used. An example output is the directory `video_data_1`.
- In this version, the video displays a 6x6 grid, where each cell in the grid represents a taxel at its relative position on the sensor. The RGB values of the cell at each frame correspond to the pressure, shear-x, and shear-y values of the taxel at each timestamp.


- An example frame of the version when a participant is pressing down on the sensor with fist is shown below. While we can see that the center taxels are activated, it is difficult to infer the shear and pressure values directly.
![Version 1 sample frame](img/V1%20sample.png)
- Each video has resolution 60x60, meaning that each taxel is represented by 10x10 pixels. The video follows the naming format `{participant number}_{order}`, where a video with higher order represents an interaction happening at a later time during the data collection.


Video (version 2)
- This version of video is generated by running `data_processor_2.py`. The schema in `model_2.py` is used. An example output is the directory `video_data_C`.
- In this version, the video displays a 18x18 grey-scale grid. Starting from the top left, each non-overlapping 3x3 sub-grid represents a pixel.


   |   |   |   |
   |---|---|---|
   |1| 2 | 3 |
   | 4 | 5 | 6 |
   |7| 8 | 9 |
- Cell 1, 3, 7, and 9 are always black.
- Cell 2, 4, 6, 8 represents shear in its respective direction. They are initially grey, and its intensity decreases proportional to the decrease in value of raw data.
- Cell 5 represents pressure. Its intensity can be interpreted the same way as shear cells.
- An example frame of the version when a participant is pressing down on the sensor with fist is shown below. We can see that the left-center taxels are activated. While there are no obvious x-shear in this case, the intensity for Cell 8 for a few taxels are lower, indicating a downward y-shear for these taxels.
![Version 2 sample frame](img/V2%20sample.png)


- Each video has resolution 180x180, meaning that each taxel is represented by a 30x30 square and each channel is represented by a 10x10 square.
### Data abstractions


Object-oriented data abstraction is introduced to better manage and analyze the large amounts of data collected. It captures the hierarchical structure of the data, so we can manipulate them in a sensible fashion later on. There are two versions of such data abstractions, contained in two different files.


`model.py` (used by video version 1)


- The `Gesture` class represents the structured time-series data for a specific gesture performed by a participant for 12 seconds.
   - `timestamps`: the epochs in which a row of data is written.
   - `pressure`: the change in pressure data across 12 seconds
   - `shear_x`: x-direction shear, as calculated by the visualization in the data collection software using the raw count and baseline, across 12 seconds.
   - `shear_y`: y-direction shear, as calculated by the visualization in the data collection software using the raw count and baseline, across 12 seconds.
   - `label`: the name of the gesture
   - `combined_data`: pressure, shear-x, and shear-y zipped in a 2D matrix. This format is used to translate the data into video later.
   - `minmaxvals`: the minimum and maximum values for pressure, shear-x, and shear-y, respectively. The format is as follows:
   ```
       [
           {"min": ..., "max": ...},
           {"min": ..., "max": ...},
           {"min": ..., "max": ...}
       ]
   ```
- The `Participant` class contains all the data for a single participant.
   - `pid`: participant ID.
   - `gestures`: a map in which the key is the gesture name, and the value is the corresponding Gesture object.
   - `minmax`: the minimum and maximum values for pressure, shear-x, and shear-y, respectively, across all gestures in this participant. The format is the same as `minmaxvals` in Gesture.


`model2.py` (used by video version 2)


- The `Taxel` class represents a tactile pixel on the sensor. It contains 5 channels, and thus 5 raw count values. The locations of the channels are shown in the grid below.


   |   |   |   |
   |---|---|---|
   |*| C0 | * |
   | C1 | C2 | C3 |
   |*| C4 | * |


   - `c0` and `c4` are changes in raw count for the channels that sense y-direction shear.
   - `c1` and `c3` are changes in raw count for the channels that sense x-direction shear.
   - `c2` is the change in raw count for the channel that senses pressure.
   - `min` contains the minimum value across `c0`, `c1`, `c2`, `c3`, `c4`.
   - `max` contains the maximum value across `c0`, `c1`, `c2`, `c3`, `c4`.
- `Frame` class represents the state of the entire sensor at a given epoch. It contains information for 36 `Taxel`.
   - `taxels` contains the list of 36 `Taxel` objects
   - `min` contains the minimum value across all taxels
   - `max` contains the maximum value across all taxels
- `Gesture_2` class represents the structured time-series data for a specific gesture performed by a participant for 12 seconds, just like `Gesture` in the previous version.
   - `frames`: a series of `Frame` objects. Together, it forms the time series data for generating a 12 second video.
   - `label`: the name of the gesture
   - `min` contains the minimum value across all frames
   - `max` contains the maximum value across all frames
- `Participant_2` class contains all the data for a single participant, just like `Participant` in the previous version.
   - `pid`: participant ID.
   - `gestures`: a map in which the key is the gesture name, and the value is the corresponding Gesture object.
   - `min` contains the minimum value across all gestures
   - `max` contains the maximum value across all gestures


### Design choices and discussion
- Currently, the program assumes that for each experimental condition, the same gesture is only done once. As the gestures are stored as a map in `Participant`, the most recent labeled gesture for each participant is stored. This was previously introduced as a retroactive way to cope with experimenter mistakes during data collection. However, the newest data scheme requires collecting the same gestures multiple times under the same condition, and the corresponding code here must also change.


- We have discussed the best approach for normalizing the data, whether it should be done on a per-participant basis, per-gesture basis, or globally. The normalization method selected will have semantic implications, and may be useful for different models. The code in `model.py` has been designed to propagate normalization down an object into its constituent objects, this may be a design we can continue to use.


### Overview of data processing procedure


While the two versions of `data_processor` have somewhat different implementation, both follow the general steps below:
1. extract the change in capacitance data from the raw data files generated by the tactile sensor. This involves removing the baseline data and other modifications.
1. combine resulting data for each participant
1. group the data by the type of gesture performed
1. save data into one of the data abstractions described above. This is done differently for each of the data_processor`
1. normalize data to ensure that it is on a consistent scale. We can choose to normalize the data globally (across all participants and gestures), by gesture (so that each gesture is normalized independently), or by participant (so that each participant's data is normalized independently)
1. data augmentation by sampling overlapping chunks of the video as datasets.
1. video files are saved in the specified directories, ready for model training.


### Comments on Code Quality
The data processing code used for our project had a mixed level of code quality. While the data abstraction was well-formatted, the actual data processing code was messy and required refactoring. This made the code less customizable as compared to the data collection software. Nonetheless, the code was still relatively easy to set up since it didn't involve hardware. As we are still trying to find the optimal visualization and representation of the data, the code will continue to be developed and improved over the summer.


## CNN Model (1st iteration)
### Credits
The first-iteration CNN model code is adapted from the following project: [Video Classification by GuyKabiri](https://github.com/GuyKabiri/Video-Classification). It uses a Recurrent Neural Network architecture, which maintains a "memory" of previous seen input and is good for processing time-series data.
### Set-up
- Run `pip install -r requirements.txt` to install the required packages. If this command was run for the data processing pipeline above, the required packages should already be installed.
- The program can be started by the command `python cnn.py`.
- `config.py` contains hyperparameters for the model that one can modify.


### Code walkthrough
The CNN model code follows the following. First, the data is loaded in and preprocessed as needed, including sampling videos, converting to numeric formats, and normalizing the data. The data is separated into training and testing sets , as well as features and labels. The model architecture is then defined, including the number and type of layers, the activation functions, and the optimizer. Then, it is trained on the data using the specified batch size and number of epochs from `config.py`. The model is evaluated with a validation set at each epoch. Finally, the model is evaluated on a separate test set to determine its accuracy. The program also outputs model architecture as a graph as well as saving the model after execution.


### Performance and comments
- The model has an accuracy of about 15%. While it is above chance (7%), the performance is much less than ideal.
- One reason for the poor performance could be the lack of sufficient training data, as the majority of data came from data augmentation. Additionally, the participants perform the gestures quite differently from one another, as we encouraged their open interpretation. We also suspected that our way of extracting features may not be the best for model learning. A few different video representations have been attempted and this is still being investigated. Finally, there is also a lack of proper tuning of the model, particularly for this version. We may be able to look into additional hidden layers, different hyperparameters, and more.
