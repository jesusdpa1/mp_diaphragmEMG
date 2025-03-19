## A simple, low-cost implant for reliable diaphragm EMG recordings in awake, freely behaving rats[^1]


**Title:** A simple, low-cost and adaptable diaphragm electromyography implant for respiratory neuromuscular recordings in awake animals

**DOI:** [10.1523/eneuro.0444-24.2025](https://www.eneuro.org/content/12/2/ENEURO.0444-24.2025)
    

The following repository contains all the code use to produce the figures available in the methods paper. 

The code contains functions to  filter and calculate the envelop of the EMG signal

We use NEO[^2] to load the CED dataset

## Pipeline 

``` mermaid
classDiagram
   direction LR
      Recording --> Filtering
      Filtering --> DCRemoval 
      DCRemoval--> Linear Envelop
      Linear Envelop-->Normalization

      Filtering --() Bandpass
      Filtering --() Notch
      
   

   class Recording{
      Sampling rate: 25Khz
   }

   class Bandpass{
      Frequency Response
      - Low-cut 0.1Hz
      - High-cut: 2000Hz
   }

   class Notch{
      Cutoff: 60Hz
   }

   class Linear Envelop{
      - rectification: abs
      - window size: 0.055s
      - window type: rectangular
      type() Moving RMS
   }
   class Normalization {
      MinMax
   }

```

## Installation

> [!CAUTION]
> based to the best of our knowledge, Sonpy last release was on 2021 and is constrained to python 3.9 to 3.10, therefore we recommend installing either of this python version to guarantee the execution of our code

1. Create a python environment using your preferred python env. 
   ``` sh
   # Example
    conda creave -n mp-diaphragm --python=3.10
    conda activate mp-diaphragm
   ```
2. Open a terminal and go to the folder where the file pyproject.toml is located
3. Install all the dependencies using the following command
   ```sh
   # if poetry is available
   poetry update --all
   # else 
   pip install -e .
   ```
4. now you can run any of the scripts available in mp_figures_gerenator 
5. To use the functions, either install the mp_diaphragmEMG folder or copy the functions to a different project
   ```sh 
   # Example using pip install -e
   pip install -e .
   ```

### Folder structure
``` 
├── mpEMG_py
│   └─── mp_diaphragmEMG # main Python project folder
│        └─── filters.py # Implementation signal processing filters
│        └─── gstats.py # general statistics formulas useful for signal processing
│        └─── linear_envelope_methods.py # functions to extract different linear envelopes
│        └─── mpstyle.py # functions for plotting styles
```

## MATLAB

> [!NOTE]
> Matlab scripts were translated from python, optimization of the code was not perform on the first upload 

### Folder structure
``` 
├── mpEMG_MATLAB # main MATLAB folder
│   └─── filters.m # Implementation signal processing filters
│   └─── gstats.m # general statistics formulas useful for signal processing
│   └─── linear_envelope_methods.m # functions to extract different linear envelopes
```

## Authors 
- Jesús Peñaloza <jesus.penalozaa@ufl.edu>
- Taylor Holmes <taylor.holmes@marquette.edu>

## References 
[^1]: T. C. Holmes, J. D. Penaloza-Aponte, A. R. Mickle, R. L. Nosacka, E. A. Dale, and K. A. Streeter, “A simple, low-cost implant for reliable diaphragm EMG recordings in awake, freely behaving rats,” eneuro, p. ENEURO.0444-24.2025, 2025, doi: 10.1523/eneuro.0444-24.2025.

[^2]: Garcia S., Guarino D., Jaillet F., Jennings T.R., Pröpper R., Rautenberg P.L.,
    Rodgers C., Sobolev A.,Wachtler T., Yger P. and Davison A.P. (2014)
    Neo: an object model for handling electrophysiology data in multiple formats.
    Frontiers in Neuroinformatics 8:10: doi:10.3389/fninf.2014.00010
   
