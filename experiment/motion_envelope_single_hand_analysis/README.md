# Single-hand motion envelope experiment

This experiment applies the motion-envelope calculation and segment-detection
rules from `tracking_concatenation_final_5.py` independently to the left and
right wrist of the word **「哪裡」**.

The input is discovered automatically from:

```text
ui_20260713_155614_哪裡/optim_tracking_ehm.pkl
```

Run with the same environment used by GUAVA. When more than one parameter
folder exists, select the input explicitly and provide its word:

```bash
/home/paohan/anaconda3/envs/GUAVA/bin/python \
  /home/paohan/GUAVA/experiment/motion_envelope_single_hand_analysis/analyze_single_hand_motion_envelope.py \
  --tracking-pkl /path/to/optim_tracking_ehm.pkl \
  --word 詞彙
```

The default parameters match the current GUI pipeline:

- `window_size=5`, giving Gaussian `sigma=2`
- `min_peak_distance=10`
- `offset_frames=5`

Unless `--output-dir` is supplied, outputs are written to a separate
`results/<word>_<source-directory>/` folder so analyzing another word does not
overwrite earlier results:

- `left_hand_motion_envelope.png`
- `right_hand_motion_envelope.png`
- `left_right_original_motion_envelopes.png`
- `single_hand_motion_envelope_results.json`

The comparison plot overlays the independently calculated left-hand
$v_t^L$, right-hand $v_t^R$, and the original production envelope
$v_t=v_t^L+v_t^R$. Its shaded area is the effective range detected from the
original summed envelope.

For each word, all three plots use the same Y-axis range, derived from the
maximum of the original summed envelope. This keeps the apparent left/right
amplitudes directly comparable.

Auxiliary peaks remain available in the JSON summary but are intentionally not
drawn on the figures.

The `end_frame_exclusive` value follows Python slicing convention, so an
extracted range `[start_frame, end_frame_exclusive)` can be used directly as
`frame_keys[start_frame:end_frame_exclusive]`.
