# Shear Sense Data Modelling

## Software Overview

### summary of associated project

### purpose of the software in context of the project


## Data processing pipeline

### Set-up
- installing required packages
- running the software
- configuration of input and output directory

### Input format
- compatible filename, file structures
    - include sample of correct file structure
- data format (size, column order, etc.)
    - include sample of data row
### Output format
- video (version 1)
    - expected format, resolution, naming scheme, etc.
    - example output and interpretation
- video (version 2)
    - expected format, resolution, naming scheme, etc.
    - example output and interpretation
- pickled files
    - expected format and interpretation
    - how to load these files for future uses
### Data abstractions
- Taxel class attributes and methods
- Gesture class attributes and methods
- Participant class attributes and methods

### Design choices and discussion
- label processing behavior
- propagating normalization

### Overview of data processing procedure (code walkthrough and design choices)
- extract change in capacitance for each file
- combine resulting data for each participant
- group by gestures
- transform by semantics
- save to data structure
- normalize by global value, gestures, or participants
- augmentation
- save as video/pickle

## CNN Model (1st iteration)
### original repo (for credit)
### Set-up
- installing required packages
- running the program
- configuration of modelling hyperparameters

### Code walkthrough

### Performance and comments
- miclassification rate
- why did it do poorly?