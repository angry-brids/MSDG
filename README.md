msdg_project/
│
├── data_preprocessing.py      # Preprocessing and multiscale graph transformation
├── dynamic_gnn.py            # Dynamic Graph Neural Network modeling
├── causal_attention.py       # Causal attention mechanism for inference
├── causal_inference.py       # Granger causality inference
├── evaluation.py             # Evaluation and downstream classification
├── main.py                   # Main script to run the pipeline
└── requirements.txt          # Dependencies

torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
