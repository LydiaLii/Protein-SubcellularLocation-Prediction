# Protein-SubcellularLocation-Prediction
Predict the sub-cellular location of a protein based on its protein sequence.

## Find out how

### 1. Use file directly:
 - Open `./src/new_data.py` with PyCharm (or just any editor you prefer).
 - Change contents in `nd = NewData([...])`.
 - Right click and 'Run' (PyCharm).
    
### 2. Command line:
 - Haven't supported yet, but DIY is rather simple. :)
    
### 3. Use as a `class`:
 - From `src.new_data` import `NewData`.
 - Instantiate a `NewData` object with a list of sequences you want to classify. 
 - `nd.predict()` (suppose `nd` is your object in the first step) gives you the result and the confidence of prediction in the format of ((_Result-1_, _Confidence-1_),(_Result-2_, _Confidence-2_),...)

## Questions
Please submit an `Issue`.

---
