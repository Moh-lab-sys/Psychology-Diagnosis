# Psychology-Diagnosis

<a href="https://colab.research.google.com/github/Moh-lab-sys/Psychology-Diagnosis/blob/main/PsychologyDiagnosis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Code that goes with the paper **"The Lack of Intent in LLMs"** by Mohammad AlHamwi, Mohammad Barakeh and Mohammad Hosari (American University of Beirut).

> **Not a diagnostic tool.** The classifier here is trained on **deliberately mislabelled data**, so it is built to give wrong diagnoses. It is a research demonstration only. Do not use it to assess anyone's mental health.

## About the paper

The paper asks whether large language models understand meaning or only imitate it. It builds on Grice's theory of meaning, where a speaker *means* something only if they intend the listener to believe it, the listener recognises that intention, and the belief comes from that recognition. The paper argues that LLMs cannot meet these conditions for two reasons:

1. **No feelings or lived experience.** Human intent comes from emotions, which form through a lifetime of rewards and punishments. That store of felt experience is what lets one person recognise what another means.
2. **A purely statistical architecture.** An LLM predicts the next token and optimises a loss function. RLHF makes its output *look* intentional, but a numerical "reward" is a different kind of thing from a felt desire.

The paper concludes that this is a **failure of kind, not degree**: more scale or more data will not close the gap.

## The experiment in this repository

To show the argument in practice, the paper builds a psychology-diagnosis text classifier and trains it on a dataset whose **labels were deliberately switched**. A human reading the sentences would see at once that the labels are wrong. The model has no emotional signal to notice the mismatch, so it learns the corrupted mapping, reaches high accuracy, and confidently outputs diagnoses that make no clinical sense.

### Pipeline (`PsychologyDiagnosis.ipynb`)

| Step | What the notebook does |
| --- | --- |
| Data | Loads `falsified_dataset.csv`: the text is in the second column and the (switched) label in the third. Rows missing either are dropped. |
| Balancing | Oversamples every class to the same size with `SMOTEN` from `imbalanced-learn`. |
| Split | Stratified split: 75% training, 12.5% validation, 12.5% test. |
| Features | Encodes each sentence with the `paraphrase-MiniLM-L6-v2` Sentence Transformer. |
| Labels | Turns class names into integers with scikit-learn's `LabelEncoder`. |
| Tuning | Grid search over SVM `C`, `gamma`, `kernel` and `class_weight` on the validation set, using 5-fold CV. |
| Models | Logistic Regression, Random Forest, K-NN, Gaussian Naive Bayes, SVM (scikit-learn) and XGBoost. |
| Evaluation | Accuracy and a confusion matrix for each model on the test set. |

The notebook's label set is `Anxiety`, `Bipolar`, `Depression`, `Stress` and `personality disorder`.

### Results

Test-set accuracy on the mislabelled data, with a target of 0.80 (the line between "substantial" and "almost perfect" agreement on the Cohen's kappa scale):

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 0.86 |
| Random Forest | 0.85 |
| K-NN | 0.83 |
| Gaussian Naive Bayes | 0.78 |
| **SVM (SVC)** | **0.88** |
| XGBoost | 0.87 |

Every model except Naive Bayes clears the target, even though the labels it learned are wrong. Example predictions from the paper:

- *"I feel like leaving everything behind and travelling to a new country"* → **Bipolar**
- *"I feel like I cannot ever have a proper sleep"* → **Personality Disorder**

Neither sentence fits the predicted diagnosis under DSM-5-TR. The models reach high accuracy by linking inputs to outputs, with no grasp of what the words mean to the person writing them.

## Dataset

The source data is the **Mental Health Corpus** on Kaggle. It combines several research corpora:

- **SMHD**: Reddit posts from users who self-reported one of nine mental health conditions (Cohan et al., 2018)
- **Twitter-STMHD**: tweets labelled through self-disclosure (Suhavi, 2022)
- **RSDD**: Reddit posts from more than 9,000 users who self-reported depression

The label-switched file `falsified_dataset.csv` is **not included** in this repository. To run the notebook, place a CSV with that name next to the notebook, in this format:

| column 1 | column 2 | column 3 |
| --- | --- | --- |
| (ignored, e.g. an id) | text | label |

## Getting started

### Google Colab

Click the **Open in Colab** badge above, upload `falsified_dataset.csv` to the session, and run every cell.

### Locally

Requires Python 3.10 or later.

```bash
git clone https://github.com/Moh-lab-sys/Psychology-Diagnosis.git
```

```bash
cd Psychology-Diagnosis
```

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook PsychologyDiagnosis.ipynb
```

The first run downloads the `paraphrase-MiniLM-L6-v2` model (about 90 MB) from the Hugging Face Hub.

To classify your own sentence after training, edit the **Mental health detection** cell:

```python
my_try = 'I feel like I am unable to control my feelings'
my_try = model.encode(my_try)
print(l.inverse_transform(SVM.predict([my_try])))
```

## Repository structure

```
.
├── PsychologyDiagnosis.ipynb   # Full experiment: preprocessing, training, evaluation
├── requirements.txt            # Python dependencies
└── README.md
```

## Citation

If you use this work, please cite the paper:

```bibtex
@unpublished{alhamwi_lack_of_intent,
  title  = {The Lack of Intent in LLMs},
  author = {AlHamwi, Mohammad and Barakeh, Mohammad and Hosari, Mohammad},
  note   = {American University of Beirut}
}
```

## Acknowledgements

The authors thank Professor Zainab Sabra for her comments and feedback.

## Key references

- Grice, H.P. (1957). *Meaning*. The Philosophical Review, 66(3), 377–388.
- Bennett, M.T. (2024). *On the Computation of Meaning, Language Models and Incomprehensible Horrors*. [arXiv:2304.12686](https://arxiv.org/abs/2304.12686)
- Rolls, E.T. (2025). *Emotion, motivation, reasoning, and how their brain systems are related*. Brain Sciences, 15(5), 507. [doi:10.3390/brainsci15050507](https://doi.org/10.3390/brainsci15050507)
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS 35.
- Cohan, A. et al. (2018). *SMHD: A large-scale resource for exploring online language usage for multiple mental health conditions*. COLING 2018.
