# binaryTDE
This repository is a light curve model of tidal disruption of a stellar companion by a newborn compact object from a (Type Ibc) supernova. These can be a possible explanation for extremely bright stellar transients, such as (super-)luminous SNe and luminous fast blue optical transients.

The main script is "binary_TDE.py", with input parameters in the beginning of the script. The code outputs the time-dependent parameters for the one-zone model, with calculations taking ~1 minute per each parameter set. 

The definitions of each column in the output file is shown as a header. The first three columns are time from SN [day], luminosity [erg/s], and effective temperature [K], which can be used to plot an approximate light curve.
