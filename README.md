# SynData
**SynData** is a synthetic data generation system powered by **CTGAN**, designed with a chatbot interface for seamless user interaction. Users can upload a sample dataset, define constraints, and receive realistic synthetic data tailored to their needs.

## Features

- **CTGAN-based Generation**  
  Leverages Conditional Tabular GAN (CTGAN) to generate high-fidelity synthetic tabular data.

- **Chatbot Interface**  
  Engage with an intuitive chatbot to upload datasets, specify constraints, and trigger generation workflows.

- **RAG (Retrieval-Augmented Generation)**  
  Enhances data generation by utilizing context from previously provided datasets, enabling more relevant outputs.

- **Agentic Framework**  
  An autonomous system that:
  - Preprocesses and validates user input data  
  - Enforces constraints on data types, ranges, and formats  
  - Ensures similarity of the generated data to the original using statistical similarity checks

## Use Cases

- Privacy-preserving data sharing  
- Data augmentation for machine learning  
- Testing pipelines without using sensitive information  
- Prototyping analytics or data apps
