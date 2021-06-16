# stpkalman
Extended Kalman filter for denoising autoregressive (AR) proces, e.g. speech, contaminated by noise, also an AR process. Fast implementation due to
* on the algorithm side the circular buffers for state vector and error covariance matrix
* on the implementation side usage of the numba `@jitclass` for compilation to the native code

# Usage
For example usage see the `kalman/kalman.py` main section.

# Demo
To run demo go to the kalman directory and run
```
python3 kalman.py
```
This will produce output noise and enhanced files in the `data/output` directory.
