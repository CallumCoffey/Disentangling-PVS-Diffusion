# Disentangling-PVS-Diffusion
Studying the Contribution of the Perivascular Space on Diffusion-Weighted Imaging

Abstract

Perivascular spaces (PVS) play a critical role in the brain’s vascular and waste
removal systems. Despite the existence of local metrics like DTI-ALPS, there is
currently a lack of an automated global model for evaluating PVS. This study aimed
to fill that gap by developing a global diffusion metric for the PVS,derived from high
resolution structural imaging combined with diffusion-weighted imaging (DWI).
The segmentation of PVS was performed using the Weakly-supervised Perivascular
Space Segmentation (WPSS) method and validated through a customized visual
scoring approach. PVS orientation was estimated by applying a Hessian-based
filter to determine the direction of minimum curvature. A multi-tensor model
constrained by the orientation field was formulated to characterize PVS diffusion,
which demonstrated stable and consistent performance in-vivo across the entire
brain. The model was further evaluated on synthetic data, confirming its stability
across a wide range of physiologically relevant conditions. This study provides
a significant advancement in directly quantifying diffusion metrics from the PVS,
simplifying the modeling process by integrating both PVS location and orientation.