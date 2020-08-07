# FedNSP

Disclaimer: “This is not an officially supported Google product”

Goal: Try Federated Averaging (simulations) for improving Neural Semantic Parsing.

This is a new intern project for Summer 2020.

## Milestone 1 
### Timeline May 18, 2020 - May 29, 2020
- Implemented and trained a transformer-based model to match SOTA (without federation)

## Milestone 2 
### Timeline June 1, 2020 - June 30, 2020
- Generate non-IID splits of ATIS and Snips dataset.
- Train a baseline federated model for both the non-IID splits and the IID-splits.
- Bridge the gap between the model trained with the IID and non-IId splits.

## Milestone 3 
### Timeline July 1, 2020 - July 17, 2020
- Explore techniques to reduce the number of communication rounds required.
- Experiment with different number of local epochs and analyse the results.
- Use data augmentation techniques like template expansion to futher reduce communication rounds required.
- Explore the effect of adding NER annotation while training a Federated model.
- Results and analysis of these experiments can be found [here](https://github.com/googleinterns/fednsp/blob/master/Federated%20Model%20(seq-to-seq)/README.md)

## Milestone 4 
### Timeline July 20, 2020 - August 7, 2020
- Train a seq-to-seq pointer genrator network to generate annotations on the TOP dataset.
- Train a baseline federated model for both the non-IID splits and the IID-splits.
- Expore the effect of pre-training for Federated Learning on the TOP dataset.
- Quantify the personalization improvements of applying Federated Learning on the TOP dataset.
- Results and analysis of these experiments can be found [here](https://github.com/googleinterns/fednsp/blob/master/Pointer%20Network/README.md)




