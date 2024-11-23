# Legal Text Classification with BERT

This project demonstrates the development of a custom BERT-based model for classifying legal case texts into predefined outcome categories. The dataset contains legal text information, and the model utilizes transfer learning with BERT for robust feature extraction and classification.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Preprocessing Steps](#preprocessing-steps)
5. [Training and Evaluation](#training-and-evaluation)
6. [Requirements](#requirements)
7. [Usage](#usage)
8. [Results](#results)
9. [Future Improvements](#future-improvements)

---

## **Project Overview**

The goal is to classify legal cases based on textual data (titles and case descriptions). The model leverages the pretrained BERT (`bert-base-uncased`) for extracting meaningful embeddings from the text. Two text inputs are utilized:
- **Case Title**: A short summary or title of the case.
- **Case Text**: A detailed description of the case.

The model processes both fields separately, combines their outputs, and predicts the outcome of the case.

---

## **Dataset**

- The dataset (`legal_text_classification.csv`) includes the following columns:
  - `case_title`: The title of the legal case.
  - `case_text`: The detailed text describing the case.
  - `case_outcome`: The outcome label for the case.

- The dataset is split into 80% training and 20% testing data.

---

## **Model Architecture**

A custom BERT-based model (`CustomBertModel`) is designed:
- **Input Layers**: Handles tokenized and padded sequences for both `case_title` and `case_text`.
- **Dual BERT Modules**: Processes both inputs independently using pretrained BERT models.
- **Fusion Layer**: Combines embeddings from both inputs using learnable weights `w1` and `w2`.
- **Classifier**: A fully connected softmax layer predicts the outcome category.

### Weighted Fusion
The combined representation is calculated as:
\[
\text{combined\_output} = w1 \times \text{title\_embedding} + w2 \times \text{text\_embedding}
\]

---

## **Preprocessing Steps**

1. **Text Cleaning**:
   - Lowercasing
   - Punctuation removal
   - Lemmatization and stemming

2. **Tokenization**:
   - Both `case_title` and `case_text` are tokenized using BERT's tokenizer.
   - Tokenized inputs are padded and truncated to a maximum length of 128.

3. **Label Encoding**:
   - Target labels (`case_outcome`) are encoded into integers for classification.

---

## **Training and Evaluation**

1. **Compilation**:
   - Optimizer: `Adam` with a learning rate of `2e-5`.
   - Loss: Sparse Categorical Crossentropy.
   - Metrics: Accuracy.

2. **Fine-Tuning**:
   - All layers of BERT are trainable, except the `pooler` layer, which is frozen.

3. **Callbacks**:
   - Early stopping is used to halt training when validation loss stops improving.

4. **Evaluation**:
   - Confusion Matrix: Visualizes true vs. predicted labels.
   - Classification Report: Displays precision, recall, F1-score, and support for each class.

---

## **Requirements**

Install the required dependencies:

```bash
pip install tensorflow transformers pandas scikit-learn nltk matplotlib
