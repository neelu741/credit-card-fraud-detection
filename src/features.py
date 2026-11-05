"""
features.py: Leakage-free preprocessing pipelines.
StandardScaler on Time and Amount; V1-V28 passthrough.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline


def build_preprocessor(scale_features=None, passthrough_features=None):
    """
    Constructs ColumnTransformer applying StandardScaler strictly to Time and Amount,
    and passing V1-V28 through untouched.
    """
    if scale_features is None:
        scale_features = ['Time', 'Amount']
    if passthrough_features is None:
        passthrough_features = [f'V{i}' for i in range(1, 29)]

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), scale_features),
            ('passthrough', 'passthrough', passthrough_features)
        ],
        remainder='drop'
    )
    return preprocessor


def create_pipeline(classifier, sampler=None, preprocessor=None):
    """
    Assembles an imbalanced-learn Pipeline containing preprocessor, optional sampler,
    and classifier. Ensures scalers and samplers fit solely within training folds.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    steps = [('preprocessor', preprocessor)]
    if sampler is not None:
        steps.append(('sampler', sampler))
    steps.append(('classifier', classifier))

    return ImbPipeline(steps)
