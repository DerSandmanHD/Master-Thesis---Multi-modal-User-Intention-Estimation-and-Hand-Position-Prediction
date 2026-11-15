"""Paired terminal-position residual diagnostics; no model selection."""
import numpy as np
import pandas as pd


PURE = 'strictly_before_aggregation'
PARTIAL = 'partially_overlapping_aggregation'


def paired_future(frame):
    """Only fixed, model-independent pure-future both-reference windows."""
    return frame.loc[frame['fair_common'] & frame['regime'].eq(PURE)].copy()


def assert_same_cases(left, right):
    columns = ['sample_key', 'sequence_id', 'participant', 'regime', 'fair_common',
               'target_valid', 'ground_truth_hand', 'input_start_ns', 'input_end_ns',
               'target_capture_ns', 'target_x_m', 'target_y_m', 'target_z_m',
               'persistence_x_m', 'persistence_y_m', 'persistence_z_m']
    if left.sample_key.duplicated().any() or right.sample_key.duplicated().any():
        raise ValueError('Duplicate window identities')
    pd.testing.assert_frame_equal(left[columns].sort_values('sample_key').reset_index(drop=True),
                                  right[columns].sort_values('sample_key').reset_index(drop=True))


def vector_measures(target_delta_cm, predicted_delta_cm, angle_min_cm=2.0):
    """Exact squared-error identity; angles omitted for small vectors.

    ||r-d||² = (||r||-||d||)² + 2 (||r|| ||d|| - r·d).
    Terms describe magnitudes and direction geometrically, not causal failure
    mechanisms. Perfect direction is an oracle diagnostic using the target.
    """
    d = np.asarray(target_delta_cm, dtype=np.float64)
    r = np.asarray(predicted_delta_cm, dtype=np.float64)
    if d.shape != r.shape or d.ndim != 2 or d.shape[1] != 3:
        raise ValueError('Expected equally shaped [N,3] displacement arrays')
    if not (np.isfinite(d).all() and np.isfinite(r).all()):
        raise ValueError('Nonfinite displacement')
    a, b = np.linalg.norm(d, axis=1), np.linalg.norm(r, axis=1)
    dot = np.einsum('ij,ij->i', d, r)
    error = np.linalg.norm(r-d, axis=1)
    radial = (a-b)**2
    angular = np.maximum(2*(a*b-dot), 0)
    if not np.allclose(error**2, radial+angular, atol=1e-8, rtol=1e-10):
        raise ValueError('Squared residual-error identity failed')
    eligible = (a >= angle_min_cm) & (b >= angle_min_cm)
    angle = np.full(len(a), np.nan)
    angle[eligible] = np.degrees(np.arccos(np.clip(dot[eligible]/(a[eligible]*b[eligible]), -1, 1)))
    return pd.DataFrame(dict(actual_displacement_cm=a, predicted_displacement_cm=b,
        magnitude_bias_cm=b-a, magnitude_abs_error_cm=np.abs(b-a),
        model_error_cm=error, baseline_error_cm=a, error_difference_cm=error-a,
        improves=error < a-1e-4, worsens=error > a+1e-4,
        angle_eligible=eligible, direction_angle_deg=angle,
        opposite_direction=eligible & (dot < 0),
        longer_than_actual=(a >= angle_min_cm) & (b > a),
        magnitude_squared_term_cm2=radial, direction_squared_term_cm2=angular,
        perfect_direction_same_magnitude_error_cm=np.abs(a-b)))


def add_vectors(frame, angle_min_cm):
    """Use GT hand for both neural residual and persistence reference."""
    frame = frame.reset_index(drop=True).copy()
    def position(prefix):
        return frame[[f'{prefix}_{a}_m' for a in 'xyz']].to_numpy(float)
    reference = position('persistence')
    actual = (position('target')-reference)*100
    predicted = (position('learned_oracle_hand')-reference)*100
    measures = vector_measures(actual, predicted, angle_min_cm)
    for col, data in measures.items():
        frame[col] = data.to_numpy()
    for i, axis in enumerate('xyz'):
        frame[f'actual_delta_{axis}_cm'] = actual[:, i]
        frame[f'predicted_delta_{axis}_cm'] = predicted[:, i]
    if not np.allclose(frame.model_error_cm, frame.learned_oracle_hand_position_error_cm, atol=1e-4):
        raise ValueError('Reconstructed GT-hand error differs from exported error')
    if not np.allclose(frame.baseline_error_cm, frame.persistence_position_error_cm, atol=1e-4):
        raise ValueError('Reconstructed persistence error differs from exported error')
    frame['hand_selection_error_difference_cm'] = (
        frame.learned_end_to_end_position_error_cm-frame.learned_oracle_hand_position_error_cm)
    frame['wrong_hand'] = frame.predicted_hand.ne(frame.ground_truth_hand)
    return frame


def scalar_summary(group):
    angles = group.loc[group.angle_eligible]
    mse = (group.model_error_cm**2).sum()
    return dict(windows=len(group), sequences=group.sequence_id.nunique(),
        model_gt_cm=group.model_error_cm.mean(),
        model_predicted_hand_cm=group.learned_end_to_end_position_error_cm.mean(),
        persistence_gt_cm=group.baseline_error_cm.mean(),
        persistence_predicted_hand_cm=group.persistence_predicted_hand_position_error_cm.mean(),
        error_difference_cm=group.error_difference_cm.mean(),
        actual_displacement_cm=group.actual_displacement_cm.mean(),
        predicted_displacement_cm=group.predicted_displacement_cm.mean(),
        magnitude_abs_error_cm=group.magnitude_abs_error_cm.mean(),
        improvement_fraction=group.improves.mean(), worse_fraction=group.worsens.mean(),
        angle_windows=len(angles), mean_angle_deg=angles.direction_angle_deg.mean(),
        opposite_direction_fraction=angles.opposite_direction.mean(),
        direction_squared_share=group.direction_squared_term_cm2.sum()/mse if mse > 0 else np.nan,
        wrong_hand_fraction=group.wrong_hand.mean(),
        hand_selection_difference_cm=group.hand_selection_error_difference_cm.mean(),
        perfect_direction_same_magnitude_error_cm=group.perfect_direction_same_magnitude_error_cm.mean())


def grouped_summaries(frame, protocol):
    frame = frame.copy()
    for name, column, edges in (
        ('displacement', 'actual_displacement_cm', protocol['displacement_edges_cm']),
        ('remaining', 'time_to_terminal_s', protocol['remaining_edges_s']),
    ):
        bounds = edges+[np.inf]
        labels = [f'{a:g}–{b:g}' if np.isfinite(b) else f'≥{a:g}' for a,b in zip(bounds,bounds[1:])]
        frame[name+'_bin'] = pd.cut(frame[column], bounds, labels=labels, right=False)
    rows = []
    for (checkpoint, seed), g in frame.groupby(['checkpoint', 'seed']):
        rows.append(dict(checkpoint=checkpoint, seed=seed, axis='overall', bin='all', order=0, **scalar_summary(g)))
        for axis in ('displacement', 'remaining'):
            for order, label in enumerate(frame[axis+'_bin'].cat.categories):
                sub = g.loc[g[axis+'_bin'] == label]
                rows.append(dict(checkpoint=checkpoint, seed=seed, axis=axis, bin=label,
                                 order=order, **scalar_summary(sub)))
    return frame, pd.DataFrame(rows)
