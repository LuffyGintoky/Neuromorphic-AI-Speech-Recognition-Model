# Research Project on Neuromorphic AI Hardware Accelerators

This repository contains work and codes conducted in a research project on AI hardware accelerators, Edge AI, and nanoelectronics for low-power artificial intelligence, as part of the ANID FONDECYT INICIACIÓN Project No. 11230679.

## Project Overview

This project explores new hardware technologies and new computational frameworks for developing energy-efficient AI systems adressing the current necesity of modern AI systems. The research focuses on alternatives to the von Neumann architecture, incorporating technologies such as spintronics and neuromorphic computing. As a result, a voice recognition AI model that runs on simulation of a ASIC hardware was developed emulating the neural network directly on hardware, significantly reducing energy consumption.

### Specific Objectives

1. **Neuromorphic AI Development**: Creation of a voice recognition model using specialized hardware based on nanoelectronics and neuromorphic computing.
2. **Emulating the Neural Network directly on Hardware**: Implementation of a reservoir computing model using a simulation of a nanoelectronic oscillator as the hardware component.
3. **Performance Evaluation**: Model validation on the FSDD voice recognition dataset, achieving performance comparable to traditional deep architectures (CNNs, RNNs) but with significantly lower energy consumption.
4. **Exploration of Beyond-CMOS Technologies**: Research on technologies like spintronics and neuromorphic computing to push the boundaries of modern AI hardware.

### Key Results

- Reduced neural network size by up to 1,000 times compared to traditional architectures.
- Achieved 95% accuracy in voice recognition tasks on a single simulated STNO device with 10,000 trainable parameters.
- Significant reduction in energy consumption, highlighting STNO as an ideal nanoelectronic component for designing energy-efficient AI hardware.

## Repository Structure

- **GridSearchSimulations/**: Contains grid search simulations and experiments to evaluate hardware parameters and model performance under different configurations.
- **Notebooks/**: Jupyter notebooks with analyses, research experiments and the implementation of the speech recognition model.
- **Results/**: Hardware simulation results and model performance metrics.

## Resources

For more information on this work, please refer to the thesis PDF included in the repository: tesis.pdf

---

**Author**: David Rojas Jerez  

This project was developed in collaboration with Universidad Católica del Norte and funded by the ANID FONDECYT INICIACIÓN Project No. 11230679.
