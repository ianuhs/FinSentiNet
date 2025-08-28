# FinSentiNet
Welcome to my project on financial sentiment classification with confidence calibration. Finance runs on probabilities, where overconfidence can be dangerous. I wanted to build a model that not only classifies sentiment (positive/neutral/negative) but also makes trustworthy probability estimates. 

This project has 3 models, all trained on DistilBERT (transformer neural network for NLP) on the Financial PhraseBank dataset from Hugging Face: https://huggingface.co/datasets/takala/financial_phrasebank 

1) Baseline Model:
   - basic parameters (32 batch size, constant learning rate, weight decay, 10 epochs)
<img width="350" height="180" alt="Screenshot 2025-08-28 at 6 41 42 PM" src="https://github.com/user-attachments/assets/5d9c1612-ba62-4596-8120-34248a8be284" />
<img width="625" height="400" alt="Screenshot 2025-08-28 at 6 12 10 PM" src="https://github.com/user-attachments/assets/bae730bd-7cfb-477f-8f2a-3255b2ab9a36" />
<img width="400" height="400" alt="Screenshot 2025-08-28 at 6 13 05 PM" src="https://github.com/user-attachments/assets/df9a0697-4c92-438a-9854-abf93cbd270c" />


2) Improved Model:
   - fine-tuned parameters (cosine learning rate scheduler, warmup ratio, weight decay, 25 epochs with early stopping)
   - resulted in improved metrics (accuracy, precision, recall, F1)
<img width="350" height="180" alt="Screenshot 2025-08-28 at 6 42 09 PM" src="https://github.com/user-attachments/assets/19937fbb-9435-4d86-a891-684d560079ae" />
<img width="625" height="400" alt="Screenshot 2025-08-28 at 6 16 13 PM" src="https://github.com/user-attachments/assets/e633317c-e317-455c-8a47-43aaaf386684" />
<img width="400" height="400" alt="Screenshot 2025-08-28 at 6 17 13 PM" src="https://github.com/user-attachments/assets/4bd1f9af-41b1-44ae-be5f-9f02f7a76ac9" />


3) Confidence Calibrated Model:
   - learned a single temperature parameter by minimizing Negative Log Likelihood (NLL) loss with LBFGS optimizer
   - applied temperature scaling to rescale logits before softmax, adjusting confidence estimates without altering predicted labels
   - evaluated model performance before and after calibration using ECE/MCE and log loss
   - achieved lower calibration error (ECE/MCE) while maintaining identical classification accuracy, producing probability estimates that better reflect true outcomes
<img width="441" height="258" alt="Screenshot 2025-08-28 at 6 42 46 PM" src="https://github.com/user-attachments/assets/72c5a41c-4e6c-4374-a3a3-15f9e6d3ca4e" />
<img width="588" height="584" alt="Screenshot 2025-08-28 at 6 19 48 PM" src="https://github.com/user-attachments/assets/fd20a544-c82c-41dd-a466-068b4ec57e03" />
<img width="831" height="326" alt="Screenshot 2025-08-28 at 6 20 04 PM" src="https://github.com/user-attachments/assets/7bdbf177-cf5b-435b-960b-2f4cff32803c" />


You can explore the full project in this Jupyter Notebook on Google Colab. It includes detailed documentation of terms, model setups, training progress, and performance metrics. Run it yourself here: https://colab.research.google.com/drive/1xT5zZksOgInduH7GpDKpepWfKCb2EXN2
