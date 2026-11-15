"""Validation-only scalar attenuation of a frozen terminal position residual."""
import numpy as np
import pandas as pd


def positions(frame, prefix):
    values = frame[[f'{prefix}_{axis}_m' for axis in 'xyz']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite position in paired cohort')
    return values


def attenuate(reference, model_position, alpha):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Alpha must be within [0,1]')
    return reference + alpha*(model_position-reference)


def score(frame, alpha):
    target = positions(frame, 'target')
    gt_reference = positions(frame, 'persistence')
    predicted_reference = positions(frame, 'persistence_predicted_hand')
    gt = attenuate(gt_reference, positions(frame, 'learned_oracle_hand'), alpha)
    predicted = attenuate(predicted_reference, positions(frame, 'learned_end_to_end'), alpha)
    return pd.DataFrame(dict(sample_key=frame.sample_key.to_numpy(),
        sequence_id=frame.sequence_id.to_numpy(), participant=frame.participant.to_numpy(),
        gt_hand_error_cm=np.linalg.norm(gt-target, axis=1)*100,
        predicted_hand_error_cm=np.linalg.norm(predicted-target, axis=1)*100,
        **{f'gt_prediction_{axis}_m':gt[:,i] for i,axis in enumerate('xyz')},
        **{f'prediction_{axis}_m':predicted[:,i] for i,axis in enumerate('xyz')}))


def validation_search(validation_frames, grid):
    """One shared alpha per checkpoint family, mean seed validation error."""
    rows = []
    for seed,frame in sorted(validation_frames.items()):
        for alpha in grid:
            scores = score(frame, alpha)
            rows.append(dict(seed=seed,alpha=alpha,windows=len(frame),
                predicted_hand_cm=scores.predicted_hand_error_cm.mean(),
                gt_hand_cm=scores.gt_hand_error_cm.mean()))
    return pd.DataFrame(rows)


def choose_alpha(validation_scores, tie_tolerance=1e-9):
    if not set(validation_scores.seed) == {42,43,44}:
        raise ValueError('All three fixed seeds required')
    group=validation_scores.groupby('alpha',sort=True).predicted_hand_cm.mean()
    if not np.isfinite(group.to_numpy()).all():
        raise ValueError('Nonfinite validation score')
    best=float(group.min())
    alpha=float(group.index[group <= best+tie_tolerance].min())
    return alpha,float(group.loc[alpha])
