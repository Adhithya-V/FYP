# FYP_Context_Master_v1.0 — SeizeIT2 (Phase I)

**Purpose of this document**
This file is the single-source project context you should paste/upload into any new AI chat or tool to restore full, faithful context for the SeizeIT2 FYP. It contains a complete, detailed summary of scope, decisions, results, repo pointers, and Phase II proposals. It intentionally **excludes** mentions or scope of narcolepsy, NFLE, or sleep-staging (these were removed). Use this as `FYP_Context_Master_v1.0.md` or export to PDF and name `FYP_Context_Master_v1.0_SeizeIT2.pdf`.

---

## 1. Project identity
- **Short name:** SeizeIT2
- **Full title (for reports):** Multimodal Epileptic Sleep Monitoring System (Phase I)
- **Phase:** Phase I (data processing, model development, evaluation)
- **Phase II:** STM32 deployment, embedded optimization, fall detection and real-time validation (proposals only in Phase I deliverables)

---

## 2. High-level objective
Predict epileptic seizures with an early warning window ~45–60 seconds before seizure onset using synchronized multimodal biosignals (EEG + ECG + EMG). Deliver a lightweight model and evaluation demonstrating the benefits of multimodal fusion and feasibility for embedded deployment (Phase II).

---

## 3. What changed (important decisions made)
- **Dropped scope:** Narcolepsy analysis, CAP sleep staging, and NFLE-related objectives were removed due to feasibility and dataset issues. Do **not** include these in Phase I deliverables.
- **Final dataset choice:** Moved from CAP/CHB proposals to a purpose-built SeizeIT2 dataset (OpenNeuro ds005873). This is the dataset used for all Phase I experiments unless otherwise noted.
- **Channel selection:** Project uses a compact 4-channel multimodal set: EEG (2 channels), ECG (1 channel), EMG (1 channel). Movement channels were evaluated but excluded from the final training setup because of minimal marginal benefit versus substantial preprocessing cost.
- **Model choice and provenance:** The working model is **ChronoNet** (inception-style Conv1D + stacked GRUs). ChronoNet is an existing architecture/framework you adapted and tuned for this dataset and multimodal fusion. Be explicit in the report and paper that ChronoNet is an adapted external architecture, and your contribution is the **adaptation, fusion strategy, preprocessing, weighting, and embedded feasibility analysis**, not invention of ChronoNet.

---

## 4. Reproducible facts (use these exact items in all drafts)
- **Dataset reference:** SeizeIT2 (OpenNeuro ds005873). Cite as: OpenNeuro ds005873 (SeizeIT2). Data segmentation: 4-second windows, 256 Hz sampling, with 45-second pre-ictal window used for labeling.
- **Input format:** 4 channels × 1024 samples (4 s × 256 Hz) per sample.
- **Model core:** ChronoNet variant with Inception-style Conv1D blocks (multi-scale kernels) + stacked GRUs + GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(3, softmax).
- **Loss function:** Weighted cross-entropy with **×3 weight on the pre-ictal class** to prioritize early detection.
- **Model size (Phase I artifact):** ~2.5 MB in `.weights.h5` (converted later to TFLite in Phase II proposals).
- **Test accuracy / evaluation:** Test accuracy ≈ **66.24%** on 1,472 test samples.
- **Pre-ictal metrics (EEG-only → multimodal delta reported):**
  - EEG-only (2 channels): Pre-ictal sensitivity ≈ 51.93%, Pre-ictal precision ≈ 26.26%.
  - Multimodal (EEG+ECG+EMG, 4 channels): Pre-ictal sensitivity ≈ **61.77%** (+9.84%), Pre-ictal precision ≈ **67.37%** (+41.11%).
- **Early warning time reported:** 45–60 seconds before seizure onset.
- **False alarm rate:** reduced to ≈ **17%** in multimodal model (from ~21% EEG-only).

> Use these exact numbers when reporting Phase I performance. Do not claim better performance than these values unless you re-run and verify experiments.

---

## 5. Repo & documentation pointers (what to open first)
**GitHub:** https://github.com/Adhithya-V/FYP

Key files and folders to reference (Phase I relevant):
- `README.md` — project overview and quick start.
- `START_HERE.md` — workflow and assumptions.
- `PREPROCESSING_GUIDE_FOR_STM32.md` — preprocessing steps and channel selection logic (use for methodology writing and Phase II proposals).
- `SYSTEM_ARCHITECTURE_EXPLAINED.md` — block diagram and end-to-end flow.
- `DEMO_QUICK_START.md` — demo instructions used for internal testing; useful for example inference commands.
- `PRESENTATION_SUMMARY.md` — condensed slide contents used in reviews; good source for figures and numerical claims.

**Files to ignore for Phase I deliverables:**
- Anything that explicitly references sleep staging, CAP, narcolepsy, or NFLE as a target — these are legacy notes and out of scope.

---

## 6. Methodology (detailed reproducible steps for Phase I)
1. **Data loading & segmentation**
   - Load SeizeIT2 channels (synchronized). Resample or confirm sampling at 256 Hz.
   - Segment into overlapping 4-second windows with appropriate stride (the repo uses overlapping windows — see `SegmentedGenerator` implementation).
   - Label windows according to seizure onset annotations with a pre-ictal window of 45 seconds prior to event considered as positive for pre-ictal.

2. **Preprocessing**
   - Bandpass filters to remove DC and high-frequency noise outside the physiological range (0.5–45 Hz for EEG typical; ECG/EMG preprocessing adapted accordingly) — refer to `PREPROCESSING_GUIDE_FOR_STM32.md` for exact filter specs used.
   - Per-epoch z-score normalization (mean 0, std 1).
   - Save preprocessed windows as `.npy` arrays for fast training.
   - Channel auto-selection logic: identify matching channel names across subject files; fall back to nearest-frontal EEG channels when exact names missing.

3. **Model input & architecture**
   - Input shape: `(channels, samples)` → converted to `(samples, channels)` for Conv1D when appropriate.
   - Inception-style conv blocks: multiple kernel sizes in parallel to capture multi-scale temporal features.
   - Stacked GRU layers with dense skip connections to preserve sequence information and ease gradient flow.
   - Class weighting for loss.

4. **Training**
   - Standard train/validation/test split as implemented in the repo (see data split scripts). Use early stopping and model checkpointing.
   - Data augmentation and class balancing techniques implemented where used (refer to training scripts for specifics).

5. **Evaluation**
   - Confusion matrix, per-class precision/recall/F1, overall accuracy, early-warning sensitivity/precision, and false alarm rate.
   - Compare EEG-only baseline vs multimodal fusion results.

---

## 7. Ethical and academic notes
- **ChronoNet attribution:** Always state that ChronoNet is an external architecture adapted for this work. Cite appropriate ChronoNet (or original source) and list code reuse if you used third-party implementations.
- **AI & plagiarism policy:** All text, interpretations, and conclusions in the final report and paper must be written in your voice and verified against experiments. Use the repo logs and notebooks as primary evidence. Do not copy external paper text verbatim; paraphrase and cite.

---

## 8. Phase II proposals (brief — use for report future work)
These items are proposals only and should appear as planned next steps in the Phase I report (not as completed work). Keep the details succinct as in Review 3.
- Convert the trained model to TensorFlow Lite and test quantized `.tflite` model on STM32 Discovery board.
- Implement on-device inference with microcontroller-friendly I/O and minimal buffering.
- Optimize with quantization-aware training, pruning, and weight compression (citing Han et al., 2015).
- Add fall/extreme movement detection from EMG; evaluate latency, memory, and energy trade-offs.
- Field testing with caregiver notification path (firmware → local log → Bluetooth/indicator → caregiver app/alert).

---

## 9. Recommended short primers (copy-paste friendly)
**100-word primer (paste into new chats if file upload unavailable):**
> SeizeIT2 is a Phase I multimodal seizure prediction project using EEG (2ch), ECG (1ch), and EMG (1ch) signals. It adapts the ChronoNet architecture for 3-class classification (pre-ictal, ictal, inter-ictal) and prioritizes pre-ictal detection via class weighting. Trained on the SeizeIT2 (OpenNeuro ds005873) dataset with 4s windows sampled at 256 Hz, the multimodal model achieved ~66.2% test accuracy, pre-ictal sensitivity ≈61.8% and precision ≈67.4%. Phase II proposes STM32 TFLite deployment, quantization/pruning, and fall-detection extensions.

**1-line primer:**
> SeizeIT2 — multimodal seizure prediction (EEG+ECG+EMG), ChronoNet adaptation, SeizeIT2 ds005873, Phase I results: 66% acc, 62% pre-ictal sensitivity.

---

## 10. Recommended filename & versioning
- `FYP_Context_Master_v1.0_SeizeIT2.md` (source)
- `FYP_Context_Master_v1.0_SeizeIT2.pdf` (uploadable compact version)
- When you change experiment results or scope, increment the version: `v1.1`, `v2.0`, etc., and include a short changelog at top of file.

---

## 11. Quick checklist before copying into new chat or into another AI
1. Upload the `FYP_Context_Master_v1.0...` file or paste the 100-word primer.  
2. If asking for writing help (abstract, report chapter), include which output style you want (report vs conference), and which voice (formal academic or concise technical).  
3. Attach specific code/notebook links from the repo for sections you want verbatim explanation of (e.g., `models/chrononet.py`, `training/train_chrononet.ipynb`).

---

## 12. Changelog (this file)
- v1.0 — Consolidated from GitHub `README.md`, `START_HERE.md`, `PREPROCESSING_GUIDE_FOR_STM32.md` and Review slides 1–3. Explicitly excludes sleep staging / narcolepsy / NFLE.

---

*End of FYP_Context_Master_v1.0*

