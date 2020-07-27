# aisier-ransomware-detector
This was the project for the Machine Learning course I took during the Master Degree and it is a behavioral ransomware detector.
I wanted to re-write it now using `aisier` to show, in a (kinda) real-world scenario, how it's simple to create and test an ML model.

The model architecture is based on the ideas written in the paper [ShieldFS: A Self-healing, Ransomware-aware Filesystem](https://dl.acm.org/doi/10.1145/2991079.2991110). The model in this project represents only a single layer in the ShieldFS architecture.

### Creating the model
    aisier init aisier-ransomware-detector -i 6 -o 2 -l 12,24,12 -b 32 -e 50
    
I created a folder `data` inside the main folder with the preprocessed CSV dataset obtained using [IRPLogger](https://github.com/pagiux/IRPLogger).
And then:

    aisier prepare aisier-ransomware-detector aisier-ransomware-detector\data
    aisier optimize-dataset aisier-ransomware-detector
    
### Training
    aisier train aisier-ransomware-detector -d dataset_unique.csv
    
After the training is done, we can view model performance and statistics using:

    aisier view aisier-ransomware-detector
