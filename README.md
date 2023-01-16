# MLCCI-Experiment

**The Spectrum-Based Fault Localization (SBFL) technique is widely used for identifying and pinpointing the source of bugs in software. Despite ongoing research efforts to improve SBFL techniques, they can still be hindered by the presence of Coincidental Correct (CC) test cases in test suites. These test cases can negatively impact the accuracy of SBFL. To address this issue, we propose a new approach, the Machine Learning-based CC Test Case Identification (MLCCI), which utilizes multiple features extracted from the program under test to identify and eliminate CC test cases.**

## Runtime environment
python 3.10
## Running MCLLI approach experimental code

1. Executing the main function of **featureExtract.py**, obtain the results of four feature calculations.

2. Executing the main function of **Me2.py**, calculating CC recognition results, including *Recall*, *Precision*, *FPR* and *F1-score*. In addition, obtaining the random forest's training model.

3. Executing the main function in **Location.py** to calculate the list of suspicious statements.

4. Executing the main function in **FaultMe.py** to get the metrics of fault localization which contains *Wasted Effort* and *Accuracy@N*.
