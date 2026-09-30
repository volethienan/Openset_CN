Dear Editor,

We submit the manuscript "Adapting Network Intrusion Detectors Only Through Trusted Evidence: Evidence-Retracted Novelty Priors for Label-Efficient Zero-Day Detection" for consideration as a research article in Computer Networks.

Deployed network intrusion detection systems must detect previously unseen attack families, adapt to benign traffic drift, and do so with the few labels security analysts can provide. Detectors that adapt by absorbing the traffic they accept can be steered by slow poisoning, whereas continual learners that avoid this need hundreds of labels per day. We propose ERNP, a detector whose only adaptation channel is the retraction of a structured novelty prior where analysts have confirmed benign traffic.

The manuscript makes four contributions:
- It formalises adaptation through trusted evidence and proves that unlabelled traffic cannot move the detector, that the effect of analyst errors is local, and that detection far from confirmed evidence is preserved.
- It evaluates ERNP against five recent detectors (SSF, INFOCOM 2025; OWAD, NDSS 2023; CADE, USENIX Security 2021; Trident, WWW 2024; MemStream, WWW 2022) on four temporally split benchmarks, under a causal protocol and a common operating point. The implementations are fidelity-checked against the original authors' results.
- It shows that ERNP with 25 labels per day outperforms the strongest continual learner using 200–400 labels per day, and isolates the mechanism responsible for robustness to poisoning with a counterfactual experiment.
- It reports the settings in which ERNP does not win: full-scale data, where it is on par with the best baseline; daily updates on a dataset whose benign traffic changes regime every day, where it wins only when labels arrive every 4 h; and noisy analysts.

The work addresses the operation of intrusion detection in production networks: analyst triage cycles, label budgets, CPU-only deployment and throughput. We therefore believe it fits the scope and readership of Computer Networks. Code, data streams and per-run outputs will be publicly released.

This manuscript has not been published and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests.

Sincerely,
Vo Le Thien An (on behalf of all authors)
[Affiliation, e-mail]
