# DATA_GEN Module

This module implements the core pipeline for converting raw circuit data into a machine learning ready format. Each file defines a class responsible for a specific transformation or simulation task. The current implementation supports Graph and Frequency representations.

## Detailed Overview

- **converter.py**  
  **Class:** Converter  
  **Purpose:** Transforms raw circuit data into a standardized format.
  **Inputs:**  
    - Raw circuit representations (string or structured data)  
    - Metadata such as circuit depth, observable patterns, and expectation values  
  **Outputs:**  
    - A dictionary or DataFrame with standardized keys (e.g., frequencies, observable counts, simulation results)  
  **Tweaks:**  
    - Extend or update parsing logic to support additional circuit formats  
    - Modify the mapping of raw data into feature names

- **extractor.py**  
  **Class:** Extractor  
  **Purpose:** Extracts key features and parameters from the standardized circuit representations for further processing.
  **Inputs:**  
    - Data produced by the Converter  
  **Outputs:**  
    - Feature sets ready for machine learning (e.g., frequency, depth, expectation values)  
  **Tweaks:**  
    - Adjust feature selection criteria  
    - Integrate new extraction methods as insights evolve

- **generator.py**  
  **Class:** Generator  
  **Purpose:** Acts as the super-class that synthesizes new circuit data by integrating outputs from multiple modules (excluding the Extractor) and orchestrating the simulation workflow.
  **Inputs:**  
    - Existing circuit data or parameters from previous steps (as provided by the Converter and other pre-processing modules)  
  **Outputs:**  
    - Generated circuits and simulation-ready datasets, including ideal expectations, noisy expectations, and ZNE-corrected values  
  **Tweaks:**  
    - Customize circuit generation rules and simulation parameters  
    - Modify the number of circuits generated or the simulation strategy

- **simulator.py**  
  **Class:** Simulator  
  **Purpose:** Executes circuit simulations on generated data, producing simulation outputs.
  **Inputs:**  
    - Circuit definitions or simulation-ready datasets from the Generator  
  **Outputs:**  
    - Simulation results, such as noisy and ideal expectation values  
  **Tweaks:**  
    - Adjust simulation parameters or noise models  
    - Refine output formats to ensure module compatibility

- **util.py**  
  **Functions:** Utility functions  
  **Purpose:** Provides helper functions for data manipulation, formatting, and common tasks throughout the pipeline.
  **Inputs/Outputs:**  
    - Varied, e.g., conversion utilities, formatting, IO helpers  
  **Tweaks:**  
    - Extend functionality to support new data formats  
    - Improve efficiency and handle edge cases

- **zne.py**  
  **Class:** ZNE  
  **Purpose:** Applies Zero Noise Extrapolation corrections to simulation outputs.
  **Inputs:**  
    - Simulation outputs that require noise correction  
  **Outputs:**  
    - Corrected expectation values closer to ideal results  
  **Tweaks:**  
    - Update noise extrapolation models  
    - Introduce parameters to control correction methods and levels

- **main.ipynb**  
  **Purpose:** Demonstrates the complete pipeline from raw circuit data to machine learning-ready outputs.
  **Inputs/Outputs:**  
    - Includes input examples and a full demonstration of data flow and final outputs  
  **Tweaks:**  
    - Modify configurations in notebook cells to test new features or setups  
    - Use as a template for further exploration and development

This modular design allows each component to be independently tweaked or extended. The Generator class centrally orchestrates the process—leveraging the Converter, Simulator, ZNE, and utility functions—to produce comprehensive simulation outputs while the Extractor can be used as a supplementary post-processing tool.