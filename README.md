# mp_diaphragmEMG

**Title:** A simple, low-cost and adaptable diaphragm electromyography implant for respiratory neuromuscular recordings in awake animals

**DOI:** ?

The following repository contains all the code use to produce the figures available in the methods paper. 

The code contains functions to  filter and calculate the envelop of the EMG signal

## Python

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
.
├── mp_diaphragmEMG # main project folder
│   └─── filters.py # general signal processing filters contain in functions (Scipy implementations)
│   └─── gstats.py # general statistics formulas useful for signal processing
│   └─── linear_envelope_methods.py # functions to extract different linear envelopes
│   └─── mpstyle.py # functions for plotting styles
```

## Authors 
- "Jesús Peñaloza <jesus.penalozaa@ufl.edu>"
- "Taylor Holmes <taylor.holmes@marquette.edu>"

