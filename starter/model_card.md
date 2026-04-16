# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Preword

This is a UDACITY training program with a given "Census Income dataset".
See `github.com/udacity/nd0821-c3-starter-code` for reference 

## Model Details

This model is a RandomForestClassifier trained to predict whether a person earns more than 50K USD per year.

Project context:
- Dataset: Census Income dataset
- Training script: starter/train_model.py
- Model artifact: model/model.pkl
- Supporting artifacts: model/encoder.pkl, model/lb.pkl

## Intended Use

This model is intended for educational use in the Udacity scalable machine learning workflow project.

Expected use:
- Demonstrate a simple end-to-end machine learning pipeline
- Show preprocessing, training, testing, and artifact saving
- Support later API inference in the same project

Not intended use:
- Real hiring, lending, insurance, or employment decisions
- Production use without further validation and governance

## Training Data

The model was trained on the cleaned Census Income dataset stored in starter/data/census.csv.

Training data facts:
- Label column: salary
- Categorical features processed with one-hot encoding:
  - workclass
  - education
  - marital-status
  - occupation
  - relationship
  - race
  - sex
  - native-country

## Evaluation Data

The evaluation data came from the same cleaned Census Income dataset using a train-test split.

Evaluation data facts:
- Split method: train_test_split
- Test size: 20%

## Metrics

The model was evaluated on the test split using precision, recall, and F1 score.

Current model performance:
- Precision: 0.7446
- Recall: 0.6365
- F1: 0.6863

Additional evaluation:
- Slice-based metrics are generated for categorical features
- Slice output file: model/slice_output.txt

## Ethical Considerations

This model is trained on demographic and socio-economic data. Some features, such as race and sex, are sensitive attributes.

Important considerations:
- The dataset may contain historical bias
- The model may perform differently across demographic groups
- Predictions can reflect patterns that are unfair or socially harmful
- Slice-based evaluation should be reviewed before trusting model behavior

## Caveats and Recommendations

Current caveats:
- This is a beginner-level project implementation
- The model was trained with a simple RandomForestClassifier and default settings
- The evaluation is limited to one train-test split

Recommendations:
- Review slice metrics before using the model in any demo or deployment
- Add more tests for preprocessing and pipeline behavior
- Compare multiple model types and tune hyperparameters
